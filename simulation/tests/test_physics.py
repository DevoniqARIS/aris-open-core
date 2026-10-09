"""Real Bullet/optical integration tests. No mocked plant or perception in campaigns."""
import asyncio, json, os, shutil
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.models import MissionStatus
from app.quality import read_image
from simulation.server import Manager, Start, create_app
from simulation.world import World, scene_asset
from simulation.perception import CalibrationVision

@pytest.mark.asyncio
@pytest.mark.parametrize('scenario,status,count',[
 ('clear','COMPLETE',5),('occlusion','COMPLETE',6),('blocked','INCOMPLETE',6),('battery','ABORTED',0)])
async def test_real_physics_campaign(tmp_path,scenario,status,count):
    m=Manager(tmp_path,rate=8)
    try:
        await m.begin(Start(scenario=scenario))
        await asyncio.wait_for(m.task,80)
        assert m.result['mission']['status']==status, m.result
        assert m.result['oracle']['verdict']=='PASS',m.result
        obs=m.store.observations_for_mission(m.plan.id)
        assert len(obs)==count
        assert all(o.source=='simulated_sensor' for o in obs)
        if scenario=='occlusion':
            north=[o for o in obs if o.label=='north_facade']
            assert [o.accepted for o in north]==[False,True]
            assert north[0].image_sha256!=north[1].image_sha256
            # The actual physical capture position changed, not just the result flag.
            a=m.world.capture_truth[north[0].image_sha256]['position']
            b=m.world.capture_truth[north[1].image_sha256]['position']
            assert np.linalg.norm(np.array(a)-b)>10
        path=np.array([x['p'] for x in m.world.path])
        assert len(path)>25
        assert np.max(np.linalg.norm(np.diff(path,axis=0),axis=1))<1.5
        assert (m.run_dir/'events.jsonl').is_file()
        assert (m.run_dir/'ground_truth.json').is_file()
    finally: await m.close()

@pytest.mark.asyncio
@pytest.mark.parametrize('fault',['gps','stale','link'])
async def test_runtime_sensor_faults(tmp_path,fault):
    m=Manager(tmp_path,rate=8)
    try:
        await m.begin(Start(scenario='clear'))
        while m.world.t<10:await asyncio.sleep(.03)
        m.world.inject(fault)
        await asyncio.wait_for(m.task,40)
        assert m.plan.status==MissionStatus.ABORTED
        if fault=='link':
            assert not m.plan.landed_confirmed
            assert m.plan.recovery_error
            assert m.world.telemetry().airborne
            assert m.result['oracle']['verdict']=='FAIL'
        elif fault=='stale':
            # Physics may land while ARIS correctly cannot confirm stale telemetry.
            assert not m.plan.landed_confirmed
        else:assert m.plan.landed_confirmed
    finally:await m.close()

@pytest.mark.asyncio
async def test_abort_during_actual_takeoff(tmp_path):
    m=Manager(tmp_path,rate=8)
    try:
        await m.begin(Start())
        while not m.executor or not m.executor.stops:await asyncio.sleep(.01)
        while not m.world.telemetry().airborne:await asyncio.sleep(.01)
        m.executor.request_stop(m.plan.id)
        await asyncio.wait_for(m.task,30)
        assert m.plan.status==MissionStatus.ABORTED
        assert m.plan.landed_confirmed
        assert not m.world.telemetry().airborne
    finally:await m.close()

@pytest.mark.asyncio
async def test_collision_is_reported_as_failure(tmp_path):
    m=Manager(tmp_path,rate=8)
    try:
        await m.begin(Start(scenario='collision'))
        await asyncio.wait_for(m.task,40)
        assert m.result['oracle']['collisions']
        assert m.result['oracle']['verdict']=='FAIL'
        assert m.plan.status!=MissionStatus.COMPLETE
    finally:await m.close()

def test_force_plant_does_not_teleport(tmp_path):
    w=World(tmp_path)
    try:
        before=w.state()['position']
        w.command(position=[10,10,12],airborne=True)
        assert w.state()['position']==before
        for _ in range(120):w._step()
        after=w.state()['position']
        assert .1<np.linalg.norm(np.array(after)-before)<5
        assert np.linalg.norm(np.array(after)-[10,10,12])>10
    finally:w.close()

@pytest.mark.asyncio
async def test_pixels_not_filename_determine_assessment(tmp_path):
    from app.mission_interpreter import DeterministicMissionInterpreter
    from PIL import Image
    asset=scene_asset();req=DeterministicMissionInterpreter().interpret(asset,'Inspect roof only').evidence_requirements[0]
    vision=CalibrationVision();w=World(tmp_path/'world')
    try:
        w.command(position=[0,0,12.24],heading=0,pitch=-90,airborne=True)
        for _ in range(3500):w._step()
        rgb,_=w.render(960,720)
        p=tmp_path/'roof.png';Image.fromarray(rgb).save(p)
        a=await vision.assess(p,req,asset)
        renamed=tmp_path/'wrong_name.png';shutil.copyfile(p,renamed)
        b=await vision.assess(renamed,req,asset)
        assert a.criterion_met and b.criterion_met
        Image.new('RGB',(960,720),'gray').save(p)
        c=await vision.assess(p,req,asset)
        assert not c.criterion_met
    finally:w.close()

def test_private_api_and_actual_camera(tmp_path):
    with TestClient(create_app(tmp_path,token='test-secret')) as client:
        assert client.get('/').status_code==200
        assert client.get('/demo/state').status_code==401
        headers={'Authorization':'Bearer test-secret'}
        assert client.get('/demo/state',headers=headers).status_code==200
        response=client.get('/demo/camera',headers=headers)
        assert response.status_code==200 and response.headers['content-type']=='image/jpeg'
        assert len(response.content)>1000
        assert client.post('/demo/start',headers=headers,json={'scenario':'pretend_success'}).status_code==422
        assert client.get('/demo/export',headers=headers).status_code==409

@pytest.mark.asyncio
async def test_live_vision_refuses_missing_key(tmp_path,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    m=Manager(tmp_path)
    try:
        with pytest.raises(ValueError,match='OPENAI_API_KEY'):
            await m.begin(Start(vision='openai'))
        assert m.run_id is None
    finally:await m.close()

@pytest.mark.asyncio
async def test_no_overlapping_missions(tmp_path):
    m=Manager(tmp_path,rate=8)
    try:
        await m.begin(Start())
        with pytest.raises(ValueError,match='already running'):await m.begin(Start())
    finally:await m.close()

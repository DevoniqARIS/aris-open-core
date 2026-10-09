import asyncio
from pathlib import Path
from app.maneuver import GPTManeuverInterpreter
from app.adaptive import GridReplanner

def action(kind, **kw):
    d={"kind":kind,"altitude_m":None,"speed_mps":None,"duration_s":None,"distance_m":None,
       "direction":None,"heading_deg":None,"turn_degrees":None,"target":None,"stop_distance_m":None,
       "radius_m":None,"laps":None,"orbit_direction":None,"corners":None,"corner_strategy":None,
       "inspection_targets":None,"coverage_mode":None,"coverage_strategy":None,"evidence_requirement":None,
       "standoff_m":None,"object_class":None,"object_selection":None,"discovered_inspection_style":None,"altitude_policy":None,"repeat":1}
    d.update(kw); return d

class Fake:
    async def generate_with_meta(self,instructions,content,schema,schema_name='x'):
        assert 'clean-slate' in instructions.lower()
        assert 'find_and_inspect' in instructions
        return {"disposition":"execute","summary":"Find nearest tree and photograph it","clarification_question":None,
                "assumptions":[],"actions":[action('takeoff',altitude_m=9),
                action('find_and_inspect',altitude_m=9,speed_mps=2,object_class='tree',object_selection='nearest',
                       discovered_inspection_style='photograph',evidence_requirement='clear_photo',standoff_m=7.5),
                action('return_home',speed_mps=2),action('land')]}, {"response_id":"r","model":"fake"}

def test_tree_search_is_supported_not_rejected():
    mission,_=asyncio.run(GPTManeuverInterpreter(client=Fake()).interpret(
        'Find the nearest tree, photograph it, then return home and land.',adaptive_inspection=True))
    x=next(a for a in mission.actions if a.kind=='find_and_inspect')
    assert x.object_class=='tree' and x.object_selection=='nearest'

def test_planner_starts_without_preseeded_asset():
    planner=GridReplanner()
    assert planner.obstacles == {}

def test_adaptive_world_context_does_not_advertise_preknown_asset():
    src=(Path(__file__).resolve().parents[1]/'app'/'maneuver.py').read_text()
    assert '"preknown_object_locations": [] if adaptive_inspection' in src

def test_old_explicit_inspection_actions_still_exist():
    src=(Path(__file__).resolve().parents[1]/'app'/'maneuver.py').read_text()
    assert '"inspect_asset"' in src
    assert '"inspect_corners"' in src

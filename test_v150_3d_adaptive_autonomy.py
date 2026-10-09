import asyncio
from app.adaptive import Adaptive3DPlanner
from app.maneuver import GPTManeuverInterpreter


def action(kind, **kw):
    d={"kind":kind,"altitude_m":None,"speed_mps":None,"duration_s":None,"distance_m":None,
       "direction":None,"heading_deg":None,"turn_degrees":None,"target":None,"stop_distance_m":None,
       "radius_m":None,"laps":None,"orbit_direction":None,"corners":None,"corner_strategy":None,
       "inspection_targets":None,"coverage_mode":None,"coverage_strategy":None,"evidence_requirement":None,
       "standoff_m":None,"object_class":None,"object_selection":None,"discovered_inspection_style":None,
       "altitude_policy":None,"repeat":1}
    d.update(kw); return d

class Fake:
    async def generate_with_meta(self,instructions,content,schema,schema_name='x'):
        assert 'GOAL-REGION' in instructions
        return {"disposition":"execute","summary":"Find a building and circle it safely","clarification_question":None,
                "assumptions":[],"actions":[
                    action('takeoff',altitude_m=12),
                    action('find_and_inspect',altitude_m=12,speed_mps=2,object_class='building',object_selection='nearest',
                           discovered_inspection_style='orbit',evidence_requirement='clear_photo',standoff_m=7.5,
                           altitude_policy='adaptive'),
                    action('return_home',speed_mps=2), action('land')]}, {"response_id":"r","model":"fake"}

def test_exact_natural_language_semantics_enable_3d_adaptation():
    mission,_=asyncio.run(GPTManeuverInterpreter(client=Fake()).interpret(
        'Takeoff to 12m. Find a building. Fly to it. Avoid obstacles. You are free to adjust altitude in order to avoid collisions. Circle the building. Land at home.',
        adaptive_inspection=True))
    x=next(a for a in mission.actions if a.kind=='find_and_inspect')
    assert x.object_class=='building' and x.discovered_inspection_style=='orbit'
    assert x.altitude_policy=='adaptive' and x.altitude_m==12


def wall():
    return {'name':'wall','position':[0,0,6],'size':[8,1.2,12],'top_z':12,'no_fly':False}


def test_3d_planner_climbs_over_wall_when_lateral_route_is_expensive():
    p=Adaptive3DPlanner(bounds=(-10,10,-10,10),min_altitude=5,max_altitude=20,xy_resolution=2,z_resolution=2)
    p.add_obstacles([{'name':'wide_wall','position':[0,0,6],'size':[19,1.2,12],'top_z':12,'no_fly':False}])
    route=p.plan((0,-8,9),(0,8,9),allow_altitude=True,nominal_altitude=9)
    assert any(z>13 for x,y,z in route), route
    assert all(p.segment_clear(a,b) for a,b in zip(route,route[1:]))


def test_fixed_altitude_policy_never_changes_z():
    p=Adaptive3DPlanner(bounds=(-10,10,-10,10),min_altitude=5,max_altitude=20,xy_resolution=2,z_resolution=2)
    p.add_obstacles([wall()])
    route=p.plan((0,-8,9),(0,8,9),allow_altitude=False,nominal_altitude=9)
    assert all(abs(z-9)<1e-6 for x,y,z in route)


def test_no_fly_volume_cannot_be_overflown():
    p=Adaptive3DPlanner(bounds=(-10,10,-10,10),min_altitude=5,max_altitude=20,xy_resolution=2,z_resolution=2)
    p.add_obstacles([{'name':'nf','position':[0,0,2],'size':[8,6,4],'top_z':4,'no_fly':True}])
    route=p.plan((0,-8,12),(0,8,12),allow_altitude=True,nominal_altitude=12)
    # Route must detour horizontally; a direct vertical-overflight through x=0,y=0 is invalid.
    assert any(abs(x)>5 or abs(y)>5 for x,y,z in route[1:-1])
    assert all(p.segment_clear(a,b) for a,b in zip(route,route[1:]))

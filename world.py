"""Bullet rigid-body plant + software-rendered optical sensor.

Forces, inertia, gravity and collisions are simulated. The position/yaw servo is
idealized, not Parrot firmware or a motor/aerodynamic model. No in-flight teleport.
Ground truth stays here and in the evaluator; perception receives RGB files only.
"""
import math, threading, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import cv2
import pybullet as p
from PIL import Image
from app.models import Telemetry, LatLon, Asset, ApprovedViewpoint
from app.geo import offset_m, bearing_deg

LABELS = ['north_facade', 'east_facade', 'south_facade', 'west_facade', 'roof']
ORIGIN = LatLon(lat=49.0, lon=25.0)
HOME = [-12., -12., .24]

def geo(x, y):
    return offset_m(ORIGIN, y, x)

def local(lat, lon):
    return np.array([math.radians(lon-ORIGIN.lon)*6371000*math.cos(math.radians(ORIGIN.lat)),
                     math.radians(lat-ORIGIN.lat)*6371000])

def scene_asset():
    alt = geo(17, 11)
    center = geo(0, 0)
    return Asset(id='simulation-building', name='Calibration building', height_m=4,
                 boundary=[geo(-4,-3),geo(4,-3),geo(4,3),geo(-4,3)],
                 flight_boundary=[geo(-24,-24),geo(24,-24),geo(24,24),geo(-24,24)],
                 alternate_views={'north_facade': [ApprovedViewpoint(
                     **alt.model_dump(), altitude_m=12,
                     heading_deg=bearing_deg(alt, center), gimbal_pitch_deg=-29)]})

class World:
    def __init__(self, directory, scenario='clear', seed=7, rate=1., wind=0.):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scenario, self.seed, self.rate = scenario, seed, rate
        self.wind = wind
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.closed = False
        self.client = p.connect(p.DIRECT)
        self.t = 0.
        self.target = np.array(HOME)
        self.yaw_target = 0.
        self.pitch = self.pitch_target = -35.
        self.max_speed = 2.
        self.airborne_command = False
        self.battery = 100.
        self.fault = None
        self.faults = []
        self.collisions = []
        self.path = []
        self.capture_truth = {}
        self.objects, self.markers = [], []
        self.nav_obstacles = []
        self.scene_objects = {}
        self.latest_frame = None
        p.setGravity(0,0,-9.81,physicsClientId=self.client)
        p.setTimeStep(1/120,physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=40,physicsClientId=self.client)
        self.ground = self.box('ground',[0,0,-.2],[55,55,.4],[.14,.20,.23,1])
        self.building = self.box('building',[0,0,2],[8,6,4],[.58,.64,.65,1])
        self.register_scene_object('building_primary', 'building', [0,0,2], [8,6,4], self.building)
        # Navigation geometry exists in the world, but ARIS learns it only when the
        # local obstacle sensor comes within range.
        self.nav_obstacles.append({
            # This is physical building geometry, NOT an infinite-height geofence.
            # 3D autonomy may route above it when sufficient vertical clearance exists.
            'name': 'primary_asset_keepout', 'position': [0.0,0.0,2.0],
            'size': [8.8,6.8,4.0], 'top_z': 4.0, 'no_fly': False, 'body': self.building,
        })
        # Raised trim provides genuine visual detail, independent of marker decoding.
        for z in [.8, 2, 3.2]:
            self.box('trim',[0,3.015,z],[8,.025,.08],[.24,.31,.34,1],collision=False)
            self.box('trim',[0,-3.015,z],[8,.025,.08],[.24,.31,.34,1],collision=False)
        for x in [-3,-1,1,3]:
            for y in [-3.03,3.03]:
                self.box('window',[x,y,2],[.62,.025,1.3],[.10,.23,.28,1],collision=False)
        self._markers()
        self._corner_markers()
        if scenario in {"adaptive_complex", "adaptive_changed"}:
            self._complex_site(changed=scenario == "adaptive_changed")
        if scenario in {'occlusion','blocked'}:
            width = 6.6 if scenario == 'occlusion' else 34
            self.box('occluder',[0,5.9,4],[width,.3,8],[.81,.43,.19,1])
        if scenario == 'collision':
            self.box('unmapped-obstacle',[-6,0,7],[5,.8,14],[.72,.23,.18,1])
        collision=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.32,.32,.16],physicsClientId=self.client)
        visual=p.createVisualShape(p.GEOM_BOX,halfExtents=[.32,.32,.16],rgbaColor=[.1,.9,.78,1],physicsClientId=self.client)
        self.drone=p.createMultiBody(baseMass=1.,baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,basePosition=HOME,physicsClientId=self.client)
        p.changeDynamics(self.drone,-1,linearDamping=.05,angularDamping=.9,physicsClientId=self.client)
        pos, quat = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        vel, _ = p.getBaseVelocity(self.drone, physicsClientId=self.client)
        self._state_cache = dict(position=list(pos), quaternion=list(quat), velocity=list(vel))
        self.initial_wall=time.monotonic()
        self.frozen_wall=None

    def box(self,name,center,size,color,collision=True):
        half=[v/2 for v in size]
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=half,physicsClientId=self.client) if collision else -1
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color,physicsClientId=self.client)
        body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,
                              basePosition=center,physicsClientId=self.client)
        self.objects.append(dict(name=name,position=center,size=size,color=color[:3]))
        return body

    def register_scene_object(self, object_id, object_class, center, size, body):
        self.scene_objects[object_id] = {
            'id': object_id,
            'class': object_class,
            'position': list(center),
            'size': list(size),
            'top_z': float(center[2] + size[2]/2),
            'body': body,
        }

    def scan_objects(self, position_xy, range_m=14.0):
        """Simulated onboard object detector: returns only objects within sensor range."""
        x, y = float(position_xy[0]), float(position_xy[1])
        result = []
        for obj in self.scene_objects.values():
            px, py = obj['position'][:2]
            sx, sy = obj['size'][:2]
            dx = max(abs(x-px)-sx/2, 0)
            dy = max(abs(y-py)-sy/2, 0)
            distance = math.hypot(dx,dy)
            if distance <= range_m:
                item = {k:v for k,v in obj.items() if k != 'body'}
                item['range_m'] = round(distance, 3)
                result.append(item)
        return result

    def object_visible(self, object_id):
        """Simulated object-detector visibility check from current camera pose.

        All calls into the shared PyBullet client are serialized with the physics
        worker. PyBullet's DIRECT client is native code and concurrent access from
        the asyncio/server thread and the physics thread can destabilize the process.
        """
        obj = self.scene_objects.get(object_id)
        if not obj:
            return False
        with self.lock:
            if self.closed:
                raise RuntimeError("Physics world is closed")
            if self.error:
                raise RuntimeError(self.error)
            pos, quat = p.getBasePositionAndOrientation(
                self.drone, physicsClientId=self.client
            )
            yaw = p.getEulerFromQuaternion(quat)[2]
            pitch = math.radians(self.pitch)
            eye = np.array(pos)+np.array([
                math.cos(yaw)*.5, math.sin(yaw)*.5, -.05
            ])
            center = np.array(obj['position'], dtype=float)
            ray = p.rayTest(
                eye.tolist(), center.tolist(), physicsClientId=self.client
            )[0]
            if ray[0] == obj['body']:
                return True
            # For composite/tree geometry the registered crown/body may not be the
            # first hit; a hit very near the requested object center is still detection.
            hit = np.array(ray[3]) if ray[0] >= 0 else None
            return bool(
                hit is not None and
                np.linalg.norm(hit-center) <= max(obj['size'])*.65
            )

    def nav_box(self, name, center, size, color, no_fly=False):
        body = self.box(name, center, size, color, collision=True)
        self.nav_obstacles.append({
            'name': name,
            'position': list(center),
            'size': list(size),
            'top_z': float(center[2] + size[2] / 2),
            'no_fly': bool(no_fly),
            'body': body,
        })
        return body

    def _complex_site(self, changed=False):
        b=self.nav_box('tree_sw', [-10.1, -8.8, 5.5], [2.6, 2.6, 11], [.20,.36,.20,1]); self.register_scene_object('tree_sw','tree',[-10.1,-8.8,5.5],[2.6,2.6,11],b)
        self.nav_box('south_wall', [0.0, -8.2, 6.0], [8.0, 1.2, 12], [.42,.47,.47,1])
        b=self.nav_box('tree_east', [9.6, -0.5, 6.2], [2.8, 2.8, 12.4], [.18,.38,.20,1]); self.register_scene_object('tree_east','tree',[9.6,-0.5,6.2],[2.8,2.8,12.4],b)
        self.nav_box('north_wall', [0.0, 8.2, 6.0], [8.5, 1.2, 12], [.44,.49,.50,1])
        b=self.nav_box('service_building_west', [-16.0, 5.5, 5.5], [7.0, 8.0, 11], [.34,.40,.44,1]); self.register_scene_object('service_building_west','building',[-16.0,5.5,5.5],[7.0,8.0,11],b)
        b=self.nav_box('warehouse_east', [16.0, 6.0, 5.0], [8.0, 10.0, 10], [.38,.43,.46,1]); self.register_scene_object('warehouse_east','building',[16.0,6.0,5.0],[8.0,10.0,10],b)
        b=self.nav_box('utility_tower', [11.5, 13.0, 8.0], [2.0, 2.0, 16], [.31,.34,.36,1]); self.register_scene_object('utility_tower','structure',[11.5,13.0,8.0],[2.0,2.0,16],b)
        b=self.nav_box('container_1', [-6.0, 13.0, 1.5], [6.0, 2.5, 3.0], [.47,.30,.20,1]); self.register_scene_object('container_1','structure',[-6.0,13.0,1.5],[6.0,2.5,3.0],b)
        b=self.nav_box('container_2', [6.5, -15.0, 1.5], [6.0, 2.5, 3.0], [.28,.36,.40,1]); self.register_scene_object('container_2','structure',[6.5,-15.0,1.5],[6.0,2.5,3.0],b)
        for i, (x,y) in enumerate([(-20,-5),(-18,-15),(18,-8),(20,16),(-8,19),(5,18)]):
            self.box(f'tree_trunk_{i}', [x,y,2.0], [.7,.7,4.0], [.28,.20,.13,1])
            crown = self.box(f'tree_crown_{i}', [x,y,5.5], [3.8,3.8,4.5], [.18,.34,.19,1])
            object_id=f'tree_{i}'
            self.register_scene_object(object_id,'tree',[x,y,5.0],[3.8,3.8,9.0],crown)
            self.nav_obstacles.append({'name':object_id,'position':[x,y,5.0],'size':[3.8,3.8,9.0],
                                       'top_z':9.5,'no_fly':False,'body':crown})
        self.box('road', [0,-21,.03], [52,7,.06], [.15,.17,.18,1], collision=False)
        for x in range(-22, 23, 6):
            self.box(f'road_mark_{x}', [x,-21,.07], [3,.16,.02], [.75,.72,.55,1], collision=False)
        if changed:
            self.box('new_damage_patch', [3.15, 3.045, 2.4], [1.25,.035,1.5], [.38,.08,.06,1], collision=False)

    def _corner_markers(self):
        """Two optical panels per corner, one on each adjoining face."""
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        mesh=self.directory/'corner-panel.obj'
        mesh.write_text('v -0.55 -0.55 0\nv 0.55 -0.55 0\nv 0.55 0.55 0\nv -0.55 0.55 0\n'
                        'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n')
        north=[-math.pi/2,0,0]; east=[0,math.pi/2,0]
        south=[math.pi/2,0,0]; west=[0,-math.pi/2,0]
        definitions = [
            (20,'corner_ne',[3.15,3.08,2.35],north), (21,'corner_ne',[4.08,2.15,2.35],east),
            (22,'corner_se',[4.08,-2.15,2.35],east), (23,'corner_se',[3.15,-3.08,2.35],south),
            (24,'corner_sw',[-3.15,-3.08,2.35],south), (25,'corner_sw',[-4.08,-2.15,2.35],west),
            (26,'corner_nw',[-4.08,2.15,2.35],west), (27,'corner_nw',[-3.15,3.08,2.35],north),
        ]
        for marker_id, label, pos, euler in definitions:
            grid=cv2.aruco.generateImageMarker(dictionary,marker_id,220)
            canvas=np.full((280,280),255,dtype=np.uint8);canvas[30:250,30:250]=grid
            texture=self.directory/f'marker-{marker_id}.png';Image.fromarray(canvas).save(texture)
            quat=p.getQuaternionFromEuler(euler)
            vis=p.createVisualShape(p.GEOM_MESH,fileName=str(mesh),rgbaColor=[1,1,1,1],physicsClientId=self.client)
            col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.55,.55,.006],physicsClientId=self.client)
            body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,
                     basePosition=pos,baseOrientation=quat,physicsClientId=self.client)
            tex=p.loadTexture(str(texture),physicsClientId=self.client)
            p.changeVisualShape(body,-1,textureUniqueId=tex,physicsClientId=self.client)
            self.markers.append(dict(id=marker_id,label=label,position=pos,quaternion=quat,body=body))

    def scan_obstacles(self, position_xy, range_m=10.0, altitude_m=9.0):
        """Local obstacle sensor. ARIS sees only obstacles currently within range."""
        x, y = float(position_xy[0]), float(position_xy[1])
        result = []
        for obstacle in self.nav_obstacles:
            px, py = obstacle['position'][:2]
            sx, sy = obstacle['size'][:2]
            dx = max(abs(x-px)-sx/2, 0)
            dy = max(abs(y-py)-sy/2, 0)
            distance = math.hypot(dx,dy)
            if distance <= range_m and (obstacle['no_fly'] or obstacle['top_z'] >= altitude_m + .5):
                result.append({k:v for k,v in obstacle.items() if k != 'body'})
        return result

    def _markers(self):
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        mesh=self.directory/'panel.obj'
        mesh.write_text('v -0.7 -0.7 0\nv 0.7 -0.7 0\nv 0.7 0.7 0\nv -0.7 0.7 0\n'
                        'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n')
        rotations=[[-math.pi/2,0,0],[0,math.pi/2,0],[math.pi/2,0,0],[0,-math.pi/2,0],[0,0,0]]
        for group,label in enumerate(LABELS):
            for i in range(4):
                marker_id=group*4+i
                grid=cv2.aruco.generateImageMarker(dictionary,marker_id,420)
                canvas=np.full((560,560),255,dtype=np.uint8);canvas[70:490,70:490]=grid
                texture=self.directory/f'marker-{marker_id}.png';Image.fromarray(canvas).save(texture)
                a=-1.75 if i%2==0 else 1.75
                z=1.05 if i<2 else 2.95
                if group==0: pos=[a,3.08,z]
                elif group==1: pos=[4.08,a*.65,z]
                elif group==2: pos=[a,-3.08,z]
                elif group==3: pos=[-4.08,a*.65,z]
                else: pos=[a,(-1.4 if i<2 else 1.4),4.04]
                quat=p.getQuaternionFromEuler(rotations[group])
                vis=p.createVisualShape(p.GEOM_MESH,fileName=str(mesh),rgbaColor=[1,1,1,1],physicsClientId=self.client)
                # Collision planes are thin boxes for independent visibility ray tests.
                col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.7,.7,.006],physicsClientId=self.client)
                body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,
                         basePosition=pos,baseOrientation=quat,physicsClientId=self.client)
                tex=p.loadTexture(str(texture),physicsClientId=self.client)
                p.changeVisualShape(body,-1,textureUniqueId=tex,physicsClientId=self.client)
                self.markers.append(dict(id=marker_id,label=label,position=pos,quaternion=quat,body=body))

    def start(self):
        self.thread=threading.Thread(target=self._loop,daemon=True,name='bullet-physics')
        self.thread.start()

    def _loop(self):
        deadline=time.monotonic()
        try:
            while not self.stop.is_set():
                with self.lock:
                    for _ in range(2): self._step()
                deadline+=1/60/self.rate
                self.stop.wait(max(0,deadline-time.monotonic()))
                if time.monotonic()-deadline>1: deadline=time.monotonic()
        except Exception as exc:
            self.error = f'{type(exc).__name__}: physics worker failed: {exc}'

    def _step(self):
        dt=1/120
        pos,quat=p.getBasePositionAndOrientation(self.drone,physicsClientId=self.client)
        vel,omega=p.getBaseVelocity(self.drone,physicsClientId=self.client)
        pos=np.array(pos);vel=np.array(vel)
        delta=self.target-pos
        desired=delta*1.4
        norm=np.linalg.norm(desired[:2])
        if norm>self.max_speed: desired[:2]*=self.max_speed/norm
        desired[2]=np.clip(desired[2],-1.8,1.8)
        accel=np.clip((desired-vel)*4,-5,5)
        if self.airborne_command:
            force=accel+np.array([self.wind*math.sin(self.t*.6),self.wind*.2,9.81])
            p.applyExternalForce(self.drone,-1,force.tolist(),pos.tolist(),p.WORLD_FRAME,physicsClientId=self.client)
        _,_,yaw=p.getEulerFromQuaternion(quat)
        yaw_error=(self.yaw_target-yaw+math.pi)%(2*math.pi)-math.pi
        up=np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)[:,2]
        torque=np.cross(up,[0,0,1])*1.2+np.array([0,0,yaw_error*.6])-np.array(omega)*.35
        p.applyExternalTorque(self.drone,-1,torque.tolist(),p.WORLD_FRAME,physicsClientId=self.client)
        self.pitch+=float(np.clip(self.pitch_target-self.pitch,-90*dt,90*dt))
        p.stepSimulation(physicsClientId=self.client)
        self.t+=dt
        pos_now, quat_now = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        vel_now, _ = p.getBaseVelocity(self.drone, physicsClientId=self.client)
        self._state_cache = dict(position=list(pos_now), quaternion=list(quat_now), velocity=list(vel_now))
        if self.airborne_command: self.battery=max(0,self.battery-dt*.035)
        if self.scenario=='battery' and self.t>18 and not self.faults:
            self.inject('battery')
        for c in p.getContactPoints(bodyA=self.drone,physicsClientId=self.client):
            if c[2]!=self.ground and not self.collisions:
                self.collisions.append({'sim_time':self.t,'body':c[2],'position':list(pos)})
        if int(self.t*5)!=int((self.t-dt)*5):
            self.path.append({'t':round(self.t,3),'p':list(self._state_cache['position']),'q':list(self._state_cache['quaternion'])})
            if len(self.path) > 12000:
                del self.path[:2000]

    def inject(self,fault):
        with self.lock:
            if fault=='battery': self.battery=25.
            elif fault=='wind': self.wind=1.8
            else: self.fault=fault
            self.faults.append({'kind':fault,'sim_time':self.t,'layer':'simulated_sensor' if fault!='wind' else 'external_force'})

    def state(self):
        # HTTP polling and control wait loops read a cache maintained by the physics
        # worker. This removes thousands of cross-thread native Bullet calls per demo.
        with self.lock:
            cached = self._state_cache
            return dict(position=list(cached['position']),quaternion=list(cached['quaternion']),
                        velocity=list(cached['velocity']), target=self.target.tolist(),
                        pitch=self.pitch,battery=self.battery, sim_time=self.t,
                        real_time_factor=self.t/max(.01,(self.frozen_wall or time.monotonic())-self.initial_wall),
                        collisions=list(self.collisions),faults=list(self.faults),path=list(self.path[-1500:]),
                        site_obstacle_count=len(self.nav_obstacles), backend_error=self.error)

    def telemetry(self):
        if self.error: raise RuntimeError(self.error)
        s=self.state();pt=geo(s['position'][0],s['position'][1])
        stamp=datetime.now(timezone.utc)
        if self.fault=='stale': stamp-=timedelta(seconds=10)
        return Telemetry(**pt.model_dump(),altitude_m=max(0,s['position'][2]-.24),
                 battery_percent=s['battery'],gps_ok=self.fault!='gps',link_ok=self.fault!='link',
                 airborne=s['position'][2]>.4,observed_at=stamp)

    def command(self,position=None,heading=None,pitch=None,airborne=None,speed=None):
        with self.lock:
            if self.fault=='link': raise ConnectionError('Simulated control link lost')
            if position is not None: self.target=np.array(position,dtype=float)
            if heading is not None: self.yaw_target=math.radians(90-heading)
            if pitch is not None: self.pitch_target=pitch
            if airborne is not None: self.airborne_command=airborne
            if speed is not None: self.max_speed=speed

    def render(self,width=800,height=600,truth=False,label=None):
        with self.lock:
            if self.closed:
                raise RuntimeError("Physics world is closed")
            if self.error:
                raise RuntimeError(self.error)
            pos,quat=p.getBasePositionAndOrientation(self.drone,physicsClientId=self.client)
            yaw=p.getEulerFromQuaternion(quat)[2];pitch=math.radians(self.pitch)
            forward=np.array([math.cos(yaw)*math.cos(pitch),math.sin(yaw)*math.cos(pitch),math.sin(pitch)])
            eye=np.array(pos)+np.array([math.cos(yaw)*.5,math.sin(yaw)*.5,-.05])
            up=np.array([-math.cos(yaw)*math.sin(pitch),-math.sin(yaw)*math.sin(pitch),math.cos(pitch)])
            view=p.computeViewMatrix(eye.tolist(),(eye+forward).tolist(),up.tolist())
            projection=p.computeProjectionMatrixFOV(65,width/height,.05,100)
            image=p.getCameraImage(width,height,view,projection,renderer=p.ER_TINY_RENDERER,
                    lightDirection=[-3,-4,12],shadow=0,lightAmbientCoeff=.8,lightDiffuseCoeff=.2,lightSpecularCoeff=0,
                    physicsClientId=self.client)
            rgb=np.asarray(image[2],dtype=np.uint8).reshape(height,width,4)[:,:,:3].copy()
            metadata=None
            if truth:
                # Oracle sees segmentation, the ARIS perception interface never does.
                segmentation=np.asarray(image[4]).reshape(height,width)
                pixels={str(m['id']):int((segmentation==m['body']).sum()) for m in self.markers if m['label']==label}
                visible=[]
                for m in self.markers:
                    if m['label']!=label: continue
                    matrix=np.array(p.getMatrixFromQuaternion(m['quaternion'])).reshape(3,3)
                    hits=0
                    for u in [-.45,0,.45]:
                        for v in [-.45,0,.45]:
                            pt=np.array(m['position'])+matrix@np.array([u,v,.012])
                            ray=p.rayTest(eye.tolist(),(pt+(pt-eye)*.002).tolist(),physicsClientId=self.client)[0]
                            if ray[0]==m['body']: hits+=1
                    if hits>=8 and pixels[str(m['id'])]>=80: visible.append(m['id'])
                metadata=dict(sim_time=self.t,position=list(pos),quaternion=list(quat),pitch=self.pitch,
                     target=label,visible_panels=visible,panel_pixels=pixels,criterion_met=len(visible)>=3)
            return rgb,metadata

    def close(self):
        if self.closed:
            return
        self.freeze()
        # Never disconnect a client while its worker thread may still be inside Bullet.
        if self.thread and self.thread.is_alive():
            self.error = self.error or "Physics worker did not stop cleanly; client left connected for process safety"
            return
        with self.lock:
            if not self.closed and p.isConnected(self.client):
                p.disconnect(self.client)
            self.closed = True

    def freeze(self):
        self.stop.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        self.frozen_wall=time.monotonic()

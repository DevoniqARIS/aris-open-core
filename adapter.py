import asyncio, hashlib, uuid
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from app.adapters.base import DroneAdapter
from app.adapters.capabilities import capability_profile
from app.models import Capture
from .world import local, HOME

class PhysicsDrone(DroneAdapter):
    """Mission commands change servo setpoints; Bullet alone changes aircraft pose."""
    def __init__(self,world,journal):
        self.world=world;self.journal=journal

    def capabilities(self):
        return capability_profile("physics")

    async def connect(self):
        self.journal('connected',engine='PyBullet',firmware=False)

    async def telemetry(self):
        return self.world.telemetry()

    async def _arrive(self, position, heading=None):
        """Wait for the simulated aircraft to settle with a progress watchdog.

        The old implementation could wait forever if the physics controller stopped
        converging. That made the UI look as though the backend had died. This turns
        a non-converging movement into a normal mission error that the adaptive
        executor can catch, abort and report.
        """
        loop = asyncio.get_running_loop()
        start = loop.time()
        s = self.world.state()
        initial_distance = float(np.linalg.norm(np.array(s['position']) - position))

        # Generous dynamic timeout: enough for the slowest legal movement plus settle.
        max_speed = max(0.5, float(getattr(self.world, 'max_speed', 2.0)))
        timeout_s = max(20.0, initial_distance / max_speed * 8.0 + 12.0)

        stable = 0
        best_distance = initial_distance
        last_progress = start

        while True:
            if self.world.error:
                raise RuntimeError(self.world.error)

            s = self.world.state()
            if s['collisions']:
                raise RuntimeError('Physics collision detected')

            distance = float(np.linalg.norm(np.array(s['position']) - position))
            velocity = float(np.linalg.norm(s['velocity']))

            if distance < .18 and velocity < .3:
                stable += 1
                if stable >= 3:
                    return
            else:
                stable = 0

            now = loop.time()
            if distance < best_distance - .08:
                best_distance = distance
                last_progress = now

            # If the controller has made no meaningful positional progress for a
            # sustained period, fail the command cleanly instead of hanging the server.
            if now - last_progress > 8.0 and distance > .35:
                raise TimeoutError(
                    f'Physics movement stopped converging: '
                    f'{distance:.2f} m remains to target'
                )

            if now - start > timeout_s:
                raise TimeoutError(
                    f'Physics movement timed out after {timeout_s:.1f}s: '
                    f'{distance:.2f} m remains to target'
                )

            await asyncio.sleep(.05)

    async def takeoff(self,altitude_m):
        target=[HOME[0],HOME[1],altitude_m+.24]
        self.journal('takeoff',target=target)
        self.world.command(position=target,airborne=True)
        await self._arrive(target)

    async def go_to(self,task,speed_mps):
        xy=local(task.lat,task.lon);target=[*xy,task.altitude_m+.24]
        self.journal('move',task=task.id,label=task.label,target=target,attempt=task.attempts,
                     heading=task.heading_deg,pitch=task.gimbal_pitch_deg,retry_reason=task.retry_reason)
        self.world.command(position=target,heading=task.heading_deg,pitch=task.gimbal_pitch_deg,speed=speed_mps)
        await self._arrive(target)
        # Confirm optical pose has settled; the firmware-independent sensor uses real body yaw.
        import pybullet as p, math
        for _ in range(100):
            s=self.world.state();yaw=p.getEulerFromQuaternion(s['quaternion'])[2]
            error=(math.radians(90-task.heading_deg)-yaw+math.pi)%(2*math.pi)-math.pi
            if abs(error)<.02 and abs(s['pitch']-task.gimbal_pitch_deg)<.5: return
            await asyncio.sleep(.05)
        raise TimeoutError(f'Camera attitude did not settle: yaw error {error:.4f}, pitch {s["pitch"]:.2f}/{task.gimbal_pitch_deg:.2f}')


    async def go_to_local(self,x_m,y_m,altitude_m,heading_deg,speed_mps,label="maneuver"):
        target=[float(x_m),float(y_m),float(altitude_m)+.24]
        self.journal('move_local',label=label,target=target,heading=heading_deg,speed=speed_mps)
        self.world.command(position=target,heading=heading_deg,speed=speed_mps)
        await self._arrive(target)


    async def _wait_heading(self, heading_deg, timeout_s=8.0):
        import pybullet as p, math
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            s = self.world.state()
            yaw = p.getEulerFromQuaternion(s['quaternion'])[2]
            error = (math.radians(90 - heading_deg) - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(error) < .025:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'Heading did not settle: error={math.degrees(error):.2f} deg')
            await asyncio.sleep(.03)

    async def turn_to_heading(self, heading_deg, label="turn"):
        pos = self.world.state()['position']
        self.journal('turn_local', label=label, heading=heading_deg)
        self.world.command(position=pos, heading=heading_deg)
        await self._wait_heading(heading_deg)

    async def fly_straight_local(self, x_m, y_m, altitude_m, heading_deg, speed_mps, duration_s, label="straight"):
        """Command a real timed straight segment in Bullet, then hold the achieved pose."""
        # Settle the requested heading before the timed translation begins.
        pos = self.world.state()['position']
        self.world.command(position=pos, heading=heading_deg, speed=speed_mps)
        await self._wait_heading(heading_deg)

        start = self.world.state()
        start_t = float(start['sim_time'])
        target = [float(x_m), float(y_m), float(altitude_m) + .24]
        self.journal(
            'straight_local_start',
            label=label,
            target=target,
            heading=heading_deg,
            speed=speed_mps,
            requested_duration_s=duration_s,
            start_position=start['position'],
            start_sim_time=start_t,
        )
        self.world.command(position=target, heading=heading_deg, speed=speed_mps)

        while True:
            if self.world.error:
                raise RuntimeError(self.world.error)
            state = self.world.state()
            if state['collisions']:
                raise RuntimeError('Physics collision detected')
            if float(state['sim_time']) - start_t >= float(duration_s):
                break
            await asyncio.sleep(.02)

        achieved = self.world.state()
        self.world.command(position=achieved['position'], heading=heading_deg)
        self.journal(
            'straight_local_complete',
            label=label,
            requested_duration_s=duration_s,
            achieved_duration_s=round(float(achieved['sim_time']) - start_t, 3),
            achieved_position=achieved['position'],
            projected_endpoint=target,
        )

    async def sense_obstacles(self, range_m=10.0, altitude_m=9.0):
        state = self.world.state()
        return self.world.scan_obstacles(state['position'][:2], range_m, altitude_m)

    async def sense_objects(self, range_m=14.0):
        state = self.world.state()
        return self.world.scan_objects(state['position'][:2], range_m)

    async def object_visible(self, object_id):
        # rayTest is a native Bullet call; keep it off the ASGI event loop.
        return await asyncio.to_thread(self.world.object_visible, object_id)

    async def capture_discovered(self, mission_id, object_id, label, attempt):
        rgb, _ = await asyncio.to_thread(self.world.render, 960, 720, False, None)
        path = self.world.directory / (str(uuid.uuid4()) + '.png')
        Image.fromarray(rgb).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.journal('capture', task=object_id, label=label, sha256=digest,
                     file=path.name, attempt=attempt, sim_time=self.world.state()['sim_time'])
        return {'image_path': str(path.resolve()), 'sha256': digest}

    async def orient_camera_local(self, heading_deg, pitch_deg, label="adaptive camera"):
        self.journal('camera_orient', label=label, heading=heading_deg, pitch=pitch_deg)
        pos = self.world.state()['position']
        self.world.command(position=pos, heading=heading_deg, pitch=pitch_deg)
        await self._wait_heading(heading_deg)
        for _ in range(120):
            state = self.world.state()
            if abs(state['pitch'] - pitch_deg) < .7:
                return
            await asyncio.sleep(.03)
        raise TimeoutError('Adaptive camera pitch did not settle')

    async def capture_corner(self, mission_id, corner, attempt):
        rgb, truth = await asyncio.to_thread(self.world.render, 960, 720, True, corner)
        path = self.world.directory / (str(uuid.uuid4()) + '.png')
        Image.fromarray(rgb).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.world.capture_truth[digest] = truth
        self.journal('capture', task=corner, label=corner, sha256=digest, file=path.name,
                     attempt=attempt, sim_time=(truth or {}).get('sim_time'))
        return {'image_path': str(path.resolve()), 'sha256': digest, 'truth': truth}

    async def capture_image(self,mission_id,task,attempt):
        rgb,truth=await asyncio.to_thread(self.world.render,960,720,True,task.label)
        path=self.world.directory/(str(uuid.uuid4())+'.png')
        Image.fromarray(rgb).save(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        # Side channel only for independent evaluator, never supplied to perception.
        self.world.capture_truth[digest]=truth
        self.journal('capture',task=task.id,label=task.label,sha256=digest,file=path.name,attempt=attempt,
                     sim_time=truth['sim_time'])
        return Capture(image_path=str(path.resolve()),source='simulated_sensor',
                       captured_at=datetime.now(timezone.utc),telemetry=await self.telemetry())

    async def hold(self):
        self.world.command(position=self.world.state()['position'])

    async def return_home(self):
        target=[HOME[0],HOME[1],self.world.state()['position'][2]]
        self.journal('return_home',target=target)
        self.world.command(position=target)
        await self._arrive(target)

    async def land(self):
        pos=self.world.state()['position'];target=[pos[0],pos[1],.24]
        self.journal('landing',target=target)
        self.world.command(position=target)
        await self._arrive(target)
        self.world.command(airborne=False)

    async def abort(self):
        self.journal('recovery',action='land_in_place')
        await self.hold()
        await self.land()

    async def close(self):
        self.journal('adapter_closed')

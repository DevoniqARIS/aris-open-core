from abc import ABC, abstractmethod


class DroneAdapter(ABC):
    def capabilities(self):
        from .capabilities import capability_profile
        name = self.__class__.__name__.lower()
        if "px4" in name or "mavsdk" in name:
            return capability_profile("px4")
        if "parrot" in name:
            return capability_profile("parrot")
        if "dji" in name:
            return capability_profile("dji")
        if "sim" in name or "replay" in name:
            return capability_profile("sim")
        raise ValueError(f"No capability profile for adapter class {self.__class__.__name__}")

    @abstractmethod
    async def connect(self): ...
    @abstractmethod
    async def telemetry(self): ...
    @abstractmethod
    async def takeoff(self, altitude_m): ...
    @abstractmethod
    async def go_to(self, task, speed_mps): ...
    @abstractmethod
    async def capture_image(self, mission_id, task, attempt): ...
    @abstractmethod
    async def hold(self): ...
    @abstractmethod
    async def return_home(self): ...
    @abstractmethod
    async def land(self): ...
    @abstractmethod
    async def abort(self): ...
    async def close(self):
        pass

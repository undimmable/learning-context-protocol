import asyncio
from openai import OpenAI
from collections import defaultdict
import dotenv
dotenv.load_dotenv()


openai = OpenAI()

class SemanticSignal:
    def __init__(self, name: str, payload=None):
        self.name = name
        self.payload = payload

class CountdownLatch:
    def __init__(self, required: int, on_ready):
        self.required = required
        self.on_ready = on_ready
        self._received = defaultdict(list)

    async def put(self, signal: SemanticSignal):
        self._received[signal.name].append(signal.payload)
        if len(self._received) >= self.required:
            await self.on_ready(self._received)  # fire & forget

class SideQuest:
    def __init__(self, id_, emit, coro):
        self.id = id_
        self.emit = emit
        self.coro = coro  # async func returning payload

    async def run(self, latch: CountdownLatch):
        payload = await self.coro()
        await latch.put(SemanticSignal(self.emit, payload))

class LearningContextProtocolServer:
    def __init__(self, spec):
        self.spec = spec

    async def compile_response(self, gathered):
        # 🔧 здесь будет reasoning-логика генерации ответа
        print("🌀  LATCH FIRED — compiled:", gathered)

    async def start(self):
        latch = CountdownLatch(
            required=len(self.spec["main_quest"]["emits"]),
            on_ready=self.compile_response,
        )

        # создали side-quests
        tasks = []
        for q in self.spec["side_quests"]:
            # для демо: каждая job «спит» и возвращает строку
            async def dummy_work(name=q["id"]):
                await asyncio.sleep(0.2)
                return f"payload::{name}"
            tasks.append(SideQuest(q["id"], q["emits"][0], dummy_work).run(latch))

        # main-quest эмулируем тремя подпотоками
        for sig in self.spec["main_quest"]["emits"]:
            async def main_emit(name=sig):
                await asyncio.sleep(0.1)
                return f"main::{name}"
            tasks.append(SideQuest(f"main::{sig}", sig, main_emit).run(latch))

        await asyncio.gather(*tasks)

# ---- launch demo ----
import yaml, asyncio, pathlib
spec = yaml.safe_load(pathlib.Path("tasks.yaml").read_text())
asyncio.run(LearningContextProtocolServer(spec).start())
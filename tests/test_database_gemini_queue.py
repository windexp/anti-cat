import unittest
import tempfile
from pathlib import Path
from datetime import datetime

from src.services.database import Database


class TestDatabaseGeminiQueue(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        self.db = Database(db_path=self.db_path)
        await self.db.initialize()

    async def asyncTearDown(self):
        self.tmpdir.cleanup()

    async def test_get_events_for_gemini_includes_gemini_error(self):
        now = datetime.now().isoformat()

        # pending, snapshot OK, gemini_result NULL => should be included
        await self.db.insert_event({
            "event_id": "e_pending",
            "status": "pending",
            "frigate_label": "cat",
            "frigate_data": {},
            "image_path": "data/images/x.jpg",
            "snapshot_downloaded": 1,
            "snapshot_error": None,
            "gemini_result": None,
            "final_label": None,
            "created_at": now,
            "updated_at": now,
            "error_message": None,
        })

        # gemini_error, snapshot OK, gemini_result NULL => should be included (retry)
        await self.db.insert_event({
            "event_id": "e_error",
            "status": "gemini_error",
            "frigate_label": "cat",
            "frigate_data": {},
            "image_path": "data/images/y.jpg",
            "snapshot_downloaded": 1,
            "snapshot_error": None,
            "gemini_result": None,
            "final_label": None,
            "created_at": now,
            "updated_at": now,
            "error_message": "429",
        })

        # gemini_error but gemini_result present => should NOT be included
        await self.db.insert_event({
            "event_id": "e_done",
            "status": "gemini_error",
            "frigate_label": "cat",
            "frigate_data": {},
            "image_path": "data/images/z.jpg",
            "snapshot_downloaded": 1,
            "snapshot_error": None,
            "gemini_result": {"primary_class": "cat"},
            "final_label": None,
            "created_at": now,
            "updated_at": now,
            "error_message": None,
        })

        queued = await self.db.get_events_for_gemini(limit=10)
        ids = {e["event_id"] for e in queued}

        self.assertIn("e_pending", ids)
        self.assertIn("e_error", ids)
        self.assertNotIn("e_done", ids)


if __name__ == "__main__":
    unittest.main()

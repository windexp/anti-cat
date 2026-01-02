"""SQLite 데이터베이스 관리"""
import aiosqlite
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """비동기 SQLite 데이터베이스 관리"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.db_path
        self._initialized = False
    
    async def initialize(self):
        """데이터베이스 초기화 및 테이블 생성"""
        if self._initialized:
            return
        
        # 데이터베이스 디렉토리 생성
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            # app_kv 테이블 (간단한 메타데이터 저장)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS app_kv (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT NOT NULL
                )
            """)

            # events 테이블 생성
            await db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    frigate_label TEXT NOT NULL,
                    frigate_data TEXT,
                    image_path TEXT,
                    snapshot_downloaded INTEGER DEFAULT 0,
                    snapshot_error TEXT,
                    gemini_result TEXT,
                    final_label TEXT,
                    bound_box TEXT,
                    is_bound_box INTEGER DEFAULT 0,
                    is_synthetic INTEGER DEFAULT 0,
                    source_event_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            
            # 마이그레이션: 기존 테이블에 컬럼 추가 (있으면 무시)
            await self._migrate_add_column(db, "events", "bound_box", "TEXT")
            await self._migrate_add_column(db, "events", "is_bound_box", "INTEGER DEFAULT 0")
            await self._migrate_add_column(db, "events", "is_synthetic", "INTEGER DEFAULT 0")
            await self._migrate_add_column(db, "events", "source_event_id", "TEXT")

            # Gemini 모델 레지스트리 (ListModels 결과 + 멀티모달 프로빙 결과)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS gemini_models (
                    model_name TEXT PRIMARY KEY,
                    family TEXT,
                    is_preview INTEGER DEFAULT 0,
                    is_exp INTEGER DEFAULT 0,
                    multimodal_status TEXT DEFAULT 'unknown',
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    last_probed_at TEXT,
                    last_probe_error TEXT
                )
            """)

            # family 별 대표 모델 선택
            await db.execute("""
                CREATE TABLE IF NOT EXISTS gemini_model_selection (
                    family TEXT PRIMARY KEY,
                    selected_model_name TEXT NOT NULL,
                    selection_reason TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # 인덱스 생성 (빠른 조회를 위해)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_status 
                ON events(status)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_snapshot 
                ON events(snapshot_downloaded)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_final_label 
                ON events(final_label)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_gemini_models_family
                ON gemini_models(family)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_gemini_models_multimodal
                ON gemini_models(multimodal_status)
            """)
            
            await db.commit()
        
        self._initialized = True
        logger.info(f"데이터베이스 초기화 완료: {self.db_path}")
    
    async def _migrate_add_column(self, db, table: str, column: str, column_type: str):
        """
        안전하게 컬럼 추가 (이미 있으면 무시)
        
        Args:
            db: 데이터베이스 연결
            table: 테이블명
            column: 컬럼명
            column_type: 컬럼 타입 (예: "TEXT", "INTEGER DEFAULT 0")
        """
        try:
            # 컬럼 존재 여부 확인
            cursor = await db.execute(f"PRAGMA table_info({table})")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if column not in column_names:
                logger.info(f"마이그레이션: {table}.{column} 컬럼 추가")
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
                await db.commit()
            else:
                logger.debug(f"마이그레이션: {table}.{column} 이미 존재")
        except Exception as e:
            logger.error(f"마이그레이션 실패 ({table}.{column}): {e}")
            # 치명적 에러가 아니므로 계속 진행
    
    async def event_exists(self, event_id: str) -> bool:
        """이벤트가 이미 처리되었는지 확인"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM events WHERE event_id = ? LIMIT 1",
                (event_id,)
            )
            result = await cursor.fetchone()
            return result is not None
    
    async def insert_event(self, event_data: Dict[str, Any]) -> bool:
        """새 이벤트 삽입"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO events (
                        event_id, status, frigate_label, frigate_data,
                        image_path, snapshot_downloaded, snapshot_error,
                        gemini_result, final_label,
                        created_at, updated_at, error_message,
                        is_synthetic, source_event_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_data['event_id'],
                    event_data['status'],
                    event_data['frigate_label'],
                    json.dumps(event_data.get('frigate_data', {})),
                    event_data.get('image_path'),
                    event_data.get('snapshot_downloaded', 0),
                    event_data.get('snapshot_error'),
                    json.dumps(event_data.get('gemini_result')) if event_data.get('gemini_result') else None,
                    event_data.get('final_label'),
                    event_data['created_at'],
                    event_data['updated_at'],
                    event_data.get('error_message'),
                    event_data.get('is_synthetic', 0),
                    event_data.get('source_event_id')
                ))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"이벤트 삽입 실패: {e}")
            return False
    
    async def update_event(self, event_id: str, updates: Dict[str, Any]) -> bool:
        """이벤트 업데이트"""
        try:
            # 업데이트할 필드와 값 준비
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key == 'event_id':
                    continue  # event_id는 업데이트하지 않음
                
                set_clauses.append(f"{key} = ?")
                
                # JSON 필드 처리
                if key in ['frigate_data', 'gemini_result'] and value is not None:
                    values.append(json.dumps(value))
                else:
                    values.append(value)
            
            # updated_at 자동 업데이트
            if 'updated_at' not in updates:
                set_clauses.append("updated_at = ?")
                values.append(datetime.now().isoformat())
            
            values.append(event_id)
            
            query = f"UPDATE events SET {', '.join(set_clauses)} WHERE event_id = ?"
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(query, values)
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"이벤트 업데이트 실패: {e}")
            return False
    
    async def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """특정 이벤트 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    async def get_events_by_status(self, status: str) -> List[Dict[str, Any]]:
        """상태별 이벤트 목록 조회 (스냅샷이 있는 것만)"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM events WHERE status = ? AND snapshot_downloaded = 1 ORDER BY created_at DESC",
                (status,)
            )
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    async def get_all_events(self, limit: int = 100, offset: int = 0, include_synthetic: bool = True, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """전체 이벤트 목록 조회 (스냅샷이 있는 것만, 페이지네이션)"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = "SELECT * FROM events WHERE snapshot_downloaded = 1"
            params = []
            
            if not include_synthetic:
                query += " AND (is_synthetic = 0 OR is_synthetic IS NULL)"
            
            if label:
                query += " AND final_label = ?"
                params.append(label)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = await db.execute(query, tuple(params))
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    async def get_events_count(self) -> int:
        """전체 이벤트 수"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM events")
            result = await cursor.fetchone()
            return result[0] if result else 0
    
    async def get_pending_gemini_events(self) -> List[Dict[str, Any]]:
        """Gemini 분석 대기/재시도 필요한 이벤트 조회 (스냅샷이 있는 것만)"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM events 
                WHERE status IN ('pending', 'gemini_error')
                AND snapshot_downloaded = 1
                ORDER BY created_at ASC
            """)
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    async def delete_event(self, event_id: str) -> bool:
        """이벤트 삭제"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"이벤트 삭제 실패: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """통계 조회 (스냅샷이 있는 이벤트만)"""
        async with aiosqlite.connect(self.db_path) as db:
            # 전체 이벤트 수 (스냅샷이 있는 것만)
            cursor = await db.execute("SELECT COUNT(*) FROM events WHERE snapshot_downloaded = 1")
            total = (await cursor.fetchone())[0]
            
            # 상태별 통계
            cursor = await db.execute("""
                SELECT status, COUNT(*) as count 
                FROM events
                WHERE snapshot_downloaded = 1
                GROUP BY status
            """)
            status_stats = {row[0]: row[1] for row in await cursor.fetchall()}
            
            # 라벨별 통계
            cursor = await db.execute("""
                SELECT final_label, COUNT(*) as count 
                FROM events 
                WHERE final_label IS NOT NULL AND snapshot_downloaded = 1
                GROUP BY final_label
            """)
            label_stats = {row[0]: row[1] for row in await cursor.fetchall()}
            
            # Synthetic 이벤트 수
            cursor = await db.execute("""
                SELECT COUNT(*) FROM events 
                WHERE is_synthetic = 1 AND snapshot_downloaded = 1
            """)
            synthetic_count = (await cursor.fetchone())[0]
            
            return {
                "total_images": total,
                "by_status": {
                    "pending": status_stats.get("pending", 0),
                    "classified": status_stats.get("classified", 0),
                    "mismatched": status_stats.get("mismatched", 0),
                    "gemini_error": status_stats.get("gemini_error", 0),
                    "manual_labeled": status_stats.get("manual_labeled", 0)
                },
                "by_class": {
                    "person": label_stats.get("person", 0),
                    "cat": label_stats.get("cat", 0),
                    "background": label_stats.get("background", 0),
                    "others": label_stats.get("others", 0)
                },
                "synthetic_count": synthetic_count
            }

    # === App KV ===
    async def get_kv(self, key: str) -> Optional[str]:
        """app_kv에서 값 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT value FROM app_kv WHERE key = ?",
                (key,),
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    async def set_kv(self, key: str, value: str) -> bool:
        """app_kv에 값 저장"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO app_kv(key, value, updated_at)
                    VALUES(?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (key, value, datetime.now().isoformat()),
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"app_kv 저장 실패: {e}")
            return False

    # === Gemini Model Registry ===
    async def upsert_gemini_model(
        self,
        *,
        model_name: str,
        family: Optional[str],
        is_preview: int,
        is_exp: int,
        multimodal_status: str,
        first_seen_at: str,
        last_seen_at: str,
        last_probed_at: Optional[str] = None,
        last_probe_error: Optional[str] = None,
    ) -> bool:
        """gemini_models upsert"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO gemini_models(
                        model_name, family, is_preview, is_exp,
                        multimodal_status, first_seen_at, last_seen_at,
                        last_probed_at, last_probe_error
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(model_name) DO UPDATE SET
                        family = excluded.family,
                        is_preview = excluded.is_preview,
                        is_exp = excluded.is_exp,
                        multimodal_status = excluded.multimodal_status,
                        last_seen_at = excluded.last_seen_at,
                        last_probed_at = excluded.last_probed_at,
                        last_probe_error = excluded.last_probe_error
                    """,
                    (
                        model_name,
                        family,
                        is_preview,
                        is_exp,
                        multimodal_status,
                        first_seen_at,
                        last_seen_at,
                        last_probed_at,
                        last_probe_error,
                    ),
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"gemini_models upsert 실패: {e}")
            return False

    async def get_gemini_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """gemini_models 단일 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM gemini_models WHERE model_name = ?",
                (model_name,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_gemini_models(self) -> List[Dict[str, Any]]:
        """gemini_models 전체 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM gemini_models ORDER BY family ASC, model_name ASC"
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def upsert_gemini_model_selection(
        self,
        *,
        family: str,
        selected_model_name: str,
        selection_reason: Optional[str] = None,
    ) -> bool:
        """gemini_model_selection upsert"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO gemini_model_selection(
                        family, selected_model_name, selection_reason, updated_at
                    ) VALUES(?, ?, ?, ?)
                    ON CONFLICT(family) DO UPDATE SET
                        selected_model_name = excluded.selected_model_name,
                        selection_reason = excluded.selection_reason,
                        updated_at = excluded.updated_at
                    """,
                    (
                        family,
                        selected_model_name,
                        selection_reason,
                        datetime.now().isoformat(),
                    ),
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"gemini_model_selection upsert 실패: {e}")
            return False

    async def get_selected_gemini_models(self) -> List[str]:
        """family별로 선택된 대표 모델 목록"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT selected_model_name FROM gemini_model_selection ORDER BY family ASC"
            )
            rows = await cursor.fetchall()
            return [r[0] for r in rows]

    async def get_gemini_model_selection_map(self) -> Dict[str, str]:
        """family -> selected_model_name 맵"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT family, selected_model_name FROM gemini_model_selection"
            )
            rows = await cursor.fetchall()
            return {r[0]: r[1] for r in rows}
    
    def _row_to_dict(self, row: aiosqlite.Row) -> Dict[str, Any]:
        """SQLite Row를 딕셔너리로 변환"""
        data = dict(row)
        
        # JSON 필드 파싱
        if data.get('frigate_data'):
            try:
                data['frigate_data'] = json.loads(data['frigate_data'])
            except:
                pass
        
        if data.get('gemini_result'):
            try:
                data['gemini_result'] = json.loads(data['gemini_result'])
            except:
                pass
        
        if data.get('bound_box'):
            try:
                data['bound_box'] = json.loads(data['bound_box'])
            except:
                pass
        
        return data
    
    async def get_events_without_snapshot(self, limit: int = 100) -> List[Dict[str, Any]]:
        """스냅샷이 다운로드되지 않은 이벤트 목록"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM events 
                WHERE snapshot_downloaded = 0 AND snapshot_error IS NULL
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    async def get_events_for_gemini(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Gemini 분석이 필요한 이벤트 목록 (스냅샷 O, Gemini 분석 X)"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM events 
                WHERE snapshot_downloaded = 1 
                AND gemini_result IS NULL 
                AND status IN ('pending', 'gemini_error')
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    async def get_events_without_bound_box(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        bound_box가 계산되지 않은 이벤트 목록
        (스냅샷은 있지만 is_bound_box=0 또는 NULL)
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM events 
                WHERE snapshot_downloaded = 1 
                AND (is_bound_box = 0 OR is_bound_box IS NULL)
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    async def close(self):
        """데이터베이스 연결 종료 (필요시)"""
        # aiosqlite는 컨텍스트 매니저를 사용하므로 별도 종료 불필요
        pass


# 전역 데이터베이스 인스턴스
db = Database()

import sqlite3
import json
import sys
from pathlib import Path

def export_boxes_to_json(db_path='data/events.db', output_path='data/boxes_comparison.json'):
    """
    Gemini가 판독한 이벤트의 Frigate와 Gemini bbox 정보를 JSON으로 추출
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Gemini 결과가 있는 이벤트만 조회
    query = """
    SELECT 
        event_id,
        frigate_data,
        gemini_result,
        frigate_label,
        created_at as timestamp
    FROM events
    WHERE gemini_result IS NOT NULL 
        AND gemini_result != ''
        AND gemini_result != 'null'
    ORDER BY created_at DESC
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    results = []
    
    for row in rows:
        try:
            # Frigate 데이터 파싱
            frigate_data = json.loads(row['frigate_data']) if row['frigate_data'] else {}
            data = frigate_data.get('data', {})
            
            # Gemini 결과 파싱
            gemini_result = json.loads(row['gemini_result']) if row['gemini_result'] else {}
            
            # bbox_normalized 추출 (첫 번째 것만)
            detected_objects = gemini_result.get('detected_objects', [])
            gemini_box = None
            if detected_objects and len(detected_objects) > 0:
                obj = detected_objects[0]
                if 'bbox_normalized' in obj:
                    bbox = obj['bbox_normalized']
                    # bbox_normalized는 [ymin, xmin, ymax, xmax] 배열
                    if isinstance(bbox, list) and len(bbox) == 4:
                        gemini_box = {
                            'ymin': bbox[0],
                            'xmin': bbox[1],
                            'ymax': bbox[2],
                            'xmax': bbox[3]
                        }
            
            # 결과 dict 생성
            result = {
                'event_id': row['event_id'],
                'timestamp': row['timestamp'],
                'frigate_label': row['frigate_label'],
                'gemini_label': gemini_result.get('primary_class'),
                'frigate': {
                    'box': data.get('box'),  # [x_center, y_center, width, height] (0~1)
                    'region': data.get('region')  # [x, y, width, height] (픽셀)
                },
                'gemini': {
                    'box': gemini_box  # {'ymin', 'xmin', 'ymax', 'xmax'} (0~1000) - 첫 번째 박스만
                }
            }
            
            results.append(result)
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON for event {row['event_id']}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Warning: Error processing event {row['event_id']}: {e}", file=sys.stderr)
            continue
    
    conn.close()
    
    # JSON 파일로 저장
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(results)} events to {output_path}")
    return results

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/events.db'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/boxes_comparison.json'
    
    results = export_boxes_to_json(db_path, output_path)
    
    # 샘플 출력
    if results:
        print("\nSample output (first event):")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
        print(f"\nTotal events with Gemini results: {len(results)}")

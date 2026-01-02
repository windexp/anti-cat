const { createApp, ref, computed, onMounted, watch } = Vue;

createApp({
    setup() {
        // 상태
        const stats = ref({ total_images: 0, by_class: {}, by_status: {} });
        const events = ref([]);
        const totalEvents = ref(0);
        const loading = ref(false);
        const currentTab = ref('all');
        const selectedEvent = ref(null);
        const processorRunning = ref(false);
        const offset = ref(0);
        const limit = ref(20);
        const toastVisible = ref(false);
        const toastMessage = ref('');
        const toastType = ref('success');
        const showLogModal = ref(false);
        const logs = ref([]);
        const logsLoading = ref(false);
        const modelRefreshRunning = ref(false);
        const boundBoxRecalculating = ref(false);
        const labelRegenerating = ref(false);

        const showModelSelector = ref(false);
        const modelSelectorLoading = ref(false);
        const modelFamilies = ref([]);
        const selectedFamily = ref('');
        const selectedModel = ref('');
        const isSidebarOpen = ref(true);
        const selectedLabelFilter = ref('');

        const imageResolution = ref('');

        // Template refs
        const modalImage = ref(null);
        const boxCanvas = ref(null);

        const modelOptions = computed(() => {
            const fam = modelFamilies.value.find(f => f.family === selectedFamily.value);
            return fam?.options || [];
        });

        // 탭 정의
        const tabs = computed(() => [
            { id: 'all', label: 'All Events', count: stats.value.total_images || 0, countClass: 'bg-slate-200 text-slate-700' },
            { id: 'classified', label: 'Classified', count: stats.value.by_status?.classified || 0, countClass: 'bg-emerald-100 text-emerald-700' },
            { id: 'mismatched', label: 'Mismatched', count: stats.value.by_status?.mismatched || 0, countClass: 'bg-amber-100 text-amber-700' },
            { id: 'pending', label: 'Pending', count: (stats.value.by_status?.pending || 0) + (stats.value.by_status?.gemini_error || 0), countClass: 'bg-rose-100 text-rose-700' },
            { id: 'synthetic', label: 'Synthetic', count: stats.value.synthetic_count || 0, countClass: 'bg-purple-100 text-purple-700' },
        ]);

        // API 호출
        const fetchStats = async () => {
            try {
                const res = await fetch('/api/stats');
                stats.value = await res.json();
            } catch (e) {
                console.error('통계 조회 실패:', e);
            }
        };

        const fetchStatus = async () => {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                processorRunning.value = data.is_running;
            } catch (e) {
                console.error('상태 조회 실패:', e);
            }
        };

        const fetchEvents = async () => {
            loading.value = true;
            try {
                let url = `/api/events?limit=${limit.value}&offset=${offset.value}`;
                
                // 라벨 필터 추가
                if (selectedLabelFilter.value) {
                    url += `&label=${selectedLabelFilter.value}`;
                }

                if (currentTab.value === 'classified') {
                    url = `/api/events/classified?limit=${limit.value}&offset=${offset.value}`;
                    if (selectedLabelFilter.value) {
                        url += `&label=${selectedLabelFilter.value}`;
                    }
                } else if (currentTab.value === 'mismatched') {
                    url = `/api/events/mismatched?limit=${limit.value}&offset=${offset.value}`;
                } else if (currentTab.value === 'pending') {
                    url = `/api/events/pending`;
                } else if (currentTab.value === 'synthetic') {
                    url = `/api/events/synthetic?limit=${limit.value}&offset=${offset.value}`;
                    if (selectedLabelFilter.value) {
                        url += `&label=${selectedLabelFilter.value}`;
                    }
                }

                const res = await fetch(url);
                const data = await res.json();
                events.value = data.events;
                totalEvents.value = data.total;
            } catch (e) {
                console.error('이벤트 조회 실패:', e);
            } finally {
                loading.value = false;
            }
        };

        const onFilterChange = () => {
            offset.value = 0;
            fetchEvents();
        };

        // 액션
        const openEventModal = (event) => {
            selectedEvent.value = event;
            imageResolution.value = ''; // 초기화
        };

        const onImageLoad = (event) => {
            // 이미지 해상도 저장
            const img = event.target;
            if (img) {
                imageResolution.value = `${img.naturalWidth} x ${img.naturalHeight}`;
            }
            // 이미지 로드 후 박스 그리기
            setTimeout(() => drawBoxes(), 100);
        };

        const drawBoxes = () => {
            if (!selectedEvent.value) return;

            const canvas = boxCanvas.value;
            const img = modalImage.value;

            if (!canvas || !img) {
                console.log('Canvas or image not found:', { canvas, img });
                return;
            }

            const ctx = canvas.getContext('2d');

            // 캔버스 크기를 이미지 실제 표시 크기에 맞춤
            const rect = img.getBoundingClientRect();
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            console.log('Drawing boxes:', {
                camera: selectedEvent.value.camera,
                bound_box: selectedEvent.value.bound_box,
                // frigateBox: selectedEvent.value.frigate_box,
                // frigateRegion: selectedEvent.value.frigate_region,
                // geminiBoxes: selectedEvent.value.gemini_boxes,
                canvasSize: { width: canvas.width, height: canvas.height }
            });
            if (!selectedEvent.value.bound_box || selectedEvent.value.bound_box.length !== 4) {
                console.log('No valid bound_box found');
                console.log('Selected event data:', selectedEvent.value);
                return;
            } else {
                console.log('Bound box found:', selectedEvent.value.bound_box);

                const [xCenter, yCenter, width, height] = selectedEvent.value.bound_box;
                x = Math.round((xCenter - width / 2) * canvas.width);
                y = Math.round((yCenter - height / 2) * canvas.height);
                w = Math.round(width * canvas.width);
                h = Math.round(height * canvas.height);
                console.log('Calculated bound box (pixels):', { x, y, w, h });
                                ctx.strokeStyle = 'rgba(239, 68, 68, 0.9)'; // red-500
                ctx.lineWidth = 4;
                ctx.strokeRect(x, y, w, h);

                // 라벨 배경
                ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
                const label = 'Frigate';
                const metrics = ctx.measureText(label);
                ctx.fillRect(x, y - 25, metrics.width + 10, 25);

                // 라벨 텍스트
                ctx.fillStyle = 'white';
                ctx.font = 'bold 16px sans-serif';
                ctx.fillText(label, x + 5, y - 7);
            }                
   
            // if (selectedEvent.value.gemini_boxes && Array.isArray(selectedEvent.value.gemini_boxes) && selectedEvent.value.gemini_boxes.length > 0) {
            //     console.log('Gemini boxes count:', selectedEvent.value.gemini_boxes.length);
            //     const box = selectedEvent.value.gemini_boxes[0]; // 첫 번째 박스만 가져오기

            //     if (box && box.length === 4) {
            //         const [yMin, xMin, yMax, xMax] = box;

            //         // 0~1000 scale을 픽셀로 변환
            //         const x = (xMin / 1000) * canvas.width;
            //         const y = (yMin / 1000) * canvas.height;
            //         const w = ((xMax - xMin) / 1000) * canvas.width;
            //         const h = ((yMax - yMin) / 1000) * canvas.height;

            //         console.log('Gemini box:', { yMin, xMin, yMax, xMax, x, y, w, h });

            //         ctx.strokeStyle = 'rgba(59, 130, 246, 0.9)'; // blue-500
            //         ctx.lineWidth = 4;
            //         ctx.strokeRect(x, y, w, h);

            //         // 라벨 배경
            //         ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
            //         const label = 'Gemini';
            //         const metrics = ctx.measureText(label);
            //         ctx.fillRect(x, y - 25, metrics.width + 10, 25);

            //         // 라벨 텍스트
            //         ctx.fillStyle = 'white';
            //         ctx.font = 'bold 16px sans-serif';
            //         ctx.fillText(label, x + 5, y - 7);
            //     }
            // }
            // if (selectedEvent.value.gemini_boxes && Array.isArray(selectedEvent.value.gemini_boxes)) {
            //     console.log('Gemini boxes count:', selectedEvent.value.gemini_boxes.length);
            //     selectedEvent.value.gemini_boxes.forEach((box, idx) => {
            //         if (box && box.length === 4) {
            //             const [yMin, xMin, yMax, xMax] = box;

            //             // 0~1000 scale을 픽셀로 변환
            //             const x = (xMin / 1000) * canvas.width;
            //             const y = (yMin / 1000) * canvas.height;
            //             const w = ((xMax - xMin) / 1000) * canvas.width;
            //             const h = ((yMax - yMin) / 1000) * canvas.height;

            //             console.log('Gemini box', idx, ':', { yMin, xMin, yMax, xMax, x, y, w, h });

            //             ctx.strokeStyle = 'rgba(59, 130, 246, 0.9)'; // blue-500
            //             ctx.lineWidth = 4;
            //             ctx.strokeRect(x, y, w, h);

            //             // 라벨 배경
            //             ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
            //             const label = `Gemini${idx > 0 ? ' ' + (idx + 1) : ''}`;
            //             const metrics = ctx.measureText(label);
            //             ctx.fillRect(x, y - 25, metrics.width + 10, 25);

            //             // 라벨 텍스트
            //             ctx.fillStyle = 'white';
            //             ctx.font = 'bold 16px sans-serif';
            //             ctx.fillText(label, x + 5, y - 7);
            //         }
            //     });
            // }
        };

        const labelEvent = async (label) => {
            try {
                const res = await fetch(`/api/events/${selectedEvent.value.event_id}/label`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ label })
                });

                if (res.ok) {
                    showToast('라벨이 지정되었습니다', 'success');
                    selectedEvent.value = null;
                    await fetchEvents();
                    await fetchStats();
                } else {
                    showToast('라벨 지정 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            }
        };

        const deleteEvent = async () => {
            if (!confirm('이 이벤트를 삭제하시겠습니까?')) return;

            try {
                const res = await fetch(`/api/events/${selectedEvent.value.event_id}`, {
                    method: 'DELETE'
                });

                if (res.ok) {
                    showToast('이벤트가 삭제되었습니다', 'success');
                    selectedEvent.value = null;
                    await fetchEvents();
                    await fetchStats();
                } else {
                    showToast('삭제 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            }
        };

        const retryGemini = async () => {
            try {
                const res = await fetch(`/api/events/${selectedEvent.value.event_id}/retry`, {
                    method: 'POST'
                });

                if (res.ok) {
                    const data = await res.json();
                    selectedEvent.value = data.event;
                    showToast('Gemini 분석이 재시도되었습니다', 'success');
                    await fetchEvents();
                    await fetchStats();
                } else {
                    showToast('재시도 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            }
        };

        const triggerDailyRoutine = async () => {
            try {
                showToast('일일 루틴을 시작합니다...', 'success');
                const res = await fetch('/api/process/trigger', { method: 'POST' });

                if (res.ok) {
                    showToast('일일 루틴이 완료되었습니다', 'success');
                    await fetchEvents();
                    await fetchStats();
                } else {
                    showToast('일일 루틴 실행 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            }
        };

        const refreshModels = async () => {
            if (modelRefreshRunning.value) return;
            modelRefreshRunning.value = true;
            try {
                showToast('모델 업데이트를 시작합니다...', 'success');
                const res = await fetch('/api/models/refresh', { method: 'POST' });
                const data = await res.json();

                if (res.ok) {
                    const count = Array.isArray(data.selected_models) ? data.selected_models.length : 0;
                    showToast(`모델 업데이트 완료 (대표 모델 ${count}개)`, 'success');
                } else {
                    showToast(data?.detail || '모델 업데이트 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            } finally {
                modelRefreshRunning.value = false;
            }
        };

        const loadModelSelectorData = async () => {
            modelSelectorLoading.value = true;
            try {
                const res = await fetch('/api/models/selection');
                const data = await res.json();
                if (!res.ok) {
                    showToast(data?.detail || '모델 목록 조회 실패', 'error');
                    return false;
                }
                modelFamilies.value = Array.isArray(data.families) ? data.families : [];
                if (modelFamilies.value.length === 0) {
                    showToast('선택 가능한 모델 정보가 없습니다', 'error');
                    return false;
                }

                // 기본 선택값 세팅
                if (!selectedFamily.value) {
                    selectedFamily.value = modelFamilies.value[0].family;
                }
                const fam = modelFamilies.value.find(f => f.family === selectedFamily.value) || modelFamilies.value[0];
                selectedFamily.value = fam.family;
                selectedModel.value = fam.selected_model_name || (fam.options?.[0]?.model_name || '');
                return true;
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
                return false;
            } finally {
                modelSelectorLoading.value = false;
            }
        };

        const toggleModelSelector = async () => {
            if (showModelSelector.value) {
                showModelSelector.value = false;
                return;
            }
            const ok = await loadModelSelectorData();
            if (ok) showModelSelector.value = true;
        };

        const closeModelSelector = () => {
            showModelSelector.value = false;
        };

        const reloadModelSelector = async () => {
            await loadModelSelectorData();
        };

        const onFamilyChanged = () => {
            const fam = modelFamilies.value.find(f => f.family === selectedFamily.value);
            if (!fam) return;
            selectedModel.value = fam.selected_model_name || (fam.options?.[0]?.model_name || '');
        };

        const applyModelSelection = async () => {
            try {
                modelSelectorLoading.value = true;
                const res = await fetch('/api/models/selection', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        family: selectedFamily.value,
                        selected_model_name: selectedModel.value
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    showToast(`모델 선택 완료: ${selectedFamily.value} -> ${selectedModel.value}`, 'success');
                    await loadModelSelectorData();
                    closeModelSelector();
                } else {
                    showToast(data?.detail || '모델 선택 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            } finally {
                modelSelectorLoading.value = false;
            }
        };

        const exportDataset = async () => {
            try {
                const res = await fetch('/api/export/yolo', { method: 'POST' });
                const data = await res.json();

                if (res.ok) {
                    showToast(`데이터셋 내보내기 완료: ${data.train_images + data.val_images}개`, 'success');
                } else {
                    showToast('내보내기 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            }
        };

        const showLogs = async () => {
            showLogModal.value = true;
            await fetchLogs();
        };

        const fetchLogs = async () => {
            try {
                logsLoading.value = true;
                const res = await fetch('/api/logs?lines=300');
                const data = await res.json();

                if (res.ok) {
                    logs.value = data.logs;
                } else {
                    showToast('로그 조회 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            } finally {
                logsLoading.value = false;
            }
        };

        const regenerateLabels = async () => {
            if (labelRegenerating.value) return;
            
            if (!confirm('모든 라벨 파일을 재생성하시겠습니까?')) return;
            
            labelRegenerating.value = true;
            try {
                showToast('라벨 재생성 중...', 'success');
                const res = await fetch('/api/regenerate-labels', { method: 'POST' });
                const data = await res.json();

                if (res.ok) {
                    showToast(`재생성 완료: 성공 ${data.success}개, 스킵 ${data.skipped}개, 실패 ${data.failed}개`, 'success');
                } else {
                    showToast(data?.detail || '라벨 재생성 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            } finally {
                labelRegenerating.value = false;
            }
        };

        const recalculateBoundBoxes = async () => {
            if (boundBoxRecalculating.value) return;
            
            if (!confirm('기존 이벤트들의 bound_box를 재계산하시겠습니까? (최대 100개)')) return;
            
            boundBoxRecalculating.value = true;
            try {
                showToast('bound_box 재계산 중...', 'success');
                const res = await fetch('/api/migrate/bound-boxes?limit=100', { method: 'POST' });
                const data = await res.json();

                if (res.ok) {
                    showToast(`재계산 완료: 성공 ${data.success_count}개, 실패 ${data.failed_count}개`, 'success');
                    await fetchStats();
                } else {
                    showToast(data?.detail || 'bound_box 재계산 실패', 'error');
                }
            } catch (e) {
                showToast('오류가 발생했습니다', 'error');
            } finally {
                boundBoxRecalculating.value = false;
            }
        };

        // 유틸리티
        let toastTimer = null;
        const showToast = (message, type = 'success') => {
            if (!message) return; // 빈 메시지 방지
            
            console.log(`Toast: ${message} (${type})`); // 디버깅용 로그
            
            if (toastTimer) {
                clearTimeout(toastTimer);
                toastTimer = null;
            }
            
            toastMessage.value = message;
            toastType.value = type;
            toastVisible.value = true;
            
            toastTimer = setTimeout(() => {
                toastVisible.value = false;
                toastMessage.value = ''; // 메시지 초기화
                toastTimer = null;
            }, 3000);
        };

        const formatDate = (dateStr) => {
            if (!dateStr) return '';
            const date = new Date(dateStr);
            return date.toLocaleString('ko-KR');
        };

        const formatEventTime = (eventId) => {
            if (!eventId) return '';
            
            // Synthetic ID 처리: synthetic_<source_id>_<timestamp>
            // source_id가 있으면 그것을 사용, 없으면 eventId 자체 사용
            let targetId = eventId;
            if (eventId.startsWith('synthetic_')) {
                // synthetic_ 제거
                const parts = eventId.split('_');
                if (parts.length >= 2) {
                    // synthetic_ 다음 부분이 source_id (timestamp-random)
                    // 예: synthetic_1767100464.562307-0zqscr_...
                    // parts[1]이 1767100464.562307-0zqscr
                    targetId = parts[1];
                }
            }

            // timestamp 추출 (첫 번째 - 앞부분)
            const timestampStr = targetId.split('-')[0];
            const timestamp = parseFloat(timestampStr);
            
            if (isNaN(timestamp)) return eventId; // 파싱 실패 시 ID 반환

            const date = new Date(timestamp * 1000);
            
            // 날짜와 시간 포맷팅
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            const hours = String(date.getHours()).padStart(2, '0');
            const minutes = String(date.getMinutes()).padStart(2, '0');
            const seconds = String(date.getSeconds()).padStart(2, '0');

            return `${year}-${month}-${day}<br>${hours}:${minutes}:${seconds}`;
        };

        const getLabelIcon = (label) => {
            if (!label) return 'fas fa-question';
            const l = label.toLowerCase();
            if (l === 'person') return 'fas fa-user';
            if (l === 'cat') return 'fas fa-cat';
            if (l === 'synthetic') return 'fas fa-flask';
            if (l === 'background') return 'fas fa-ban';
            return 'fas fa-tag';
        };

        const getStatusClass = (status) => {
            const classes = {
                classified: 'bg-emerald-100 text-emerald-700 border border-emerald-200',
                manual_labeled: 'bg-blue-100 text-blue-700 border border-blue-200',
                mismatched: 'bg-amber-100 text-amber-700 border border-amber-200',
                pending: 'bg-slate-100 text-slate-600 border border-slate-200',
                gemini_error: 'bg-rose-100 text-rose-700 border border-rose-200'
            };
            return classes[status] || 'bg-slate-100 text-slate-600 border border-slate-200';
        };

        const getStatusLabel = (status) => {
            const labels = {
                classified: 'Classified',
                manual_labeled: 'Manual Labeled',
                mismatched: 'Mismatched',
                pending: 'Pending',
                gemini_error: 'Error'
            };
            return labels[status] || status;
        };

        const getLabelClass = (label) => {
            const classes = {
                person: 'bg-blue-50 text-blue-600 border border-blue-100',
                cat: 'bg-orange-50 text-orange-600 border border-orange-100',
                null: 'bg-slate-50 text-slate-600 border border-slate-200',
                background: 'bg-slate-50 text-slate-600 border border-slate-200'
            };
            return classes[label] || 'bg-slate-50 text-slate-600 border border-slate-200';
        };

        const getConfidenceClass = (event) => {
            // 1) Gemini confidence 문자열 우선
            const gemini = event?.gemini_confidence;
            if (gemini === 'high') return 'bg-emerald-50 text-emerald-700 border-emerald-100';
            if (gemini === 'medium') return 'bg-amber-50 text-amber-700 border-amber-100';
            if (gemini === 'low') return 'bg-rose-50 text-rose-600 border-rose-100';

            // 2) 숫자 점수(0~1)면 임계값으로 매핑
            const score = event?.frigate_confidence;
            if (typeof score === 'number' && !Number.isNaN(score)) {
                if (score >= 0.8) return 'bg-emerald-50 text-emerald-700 border-emerald-100';
                if (score >= 0.7) return 'bg-amber-50 text-amber-700 border-amber-100';
                return 'bg-rose-50 text-rose-600 border-rose-100';
            }

            // 값이 없으면 회색
            return 'bg-slate-50 text-slate-600 border-slate-200';
        };

        const getFrigateConfidenceClass = (event) => {
            const score = event?.frigate_confidence;
            if (typeof score === 'number' && !Number.isNaN(score)) {
                if (score >= 0.8) return 'bg-emerald-50 text-emerald-700 border-emerald-100';
                if (score >= 0.7) return 'bg-amber-50 text-amber-700 border-amber-100';
                return 'bg-rose-50 text-rose-600 border-rose-100';
            }
            // 값이 없으면 회색
            return 'bg-slate-50 text-slate-600 border-slate-200';
        };

        const prevPage = () => {
            if (offset.value > 0) {
                offset.value = Math.max(0, offset.value - limit.value);
            }
        };

        const nextPage = () => {
            if (offset.value + limit.value < totalEvents.value) {
                offset.value += limit.value;
            }
        };

        // 탭 변경 시 데이터 새로고침
        watch(currentTab, () => {
            offset.value = 0;
            selectedLabelFilter.value = '';
            fetchEvents();
        });

        watch(offset, () => {
            fetchEvents();
        });

        // selectedEvent 변경 시 박스 다시 그리기
        watch(selectedEvent, () => {
            if (selectedEvent.value) {
                setTimeout(() => drawBoxes(), 100);
            }
        });

        // 초기 로드
        onMounted(() => {
            console.log('App mounted');
            // 토스트 상태 초기화
            toastVisible.value = false;
            toastMessage.value = '';
            
            fetchStats();
            fetchStatus();
            fetchEvents();

            // 주기적으로 통계 업데이트
            setInterval(() => {
                fetchStats();
                fetchStatus();
            }, 30000);
        });

        return {
            stats, events, totalEvents, loading, currentTab, selectedEvent,
            processorRunning, offset, limit, toastVisible, toastMessage, toastType, tabs,
            showLogModal, logs, logsLoading,
            modelRefreshRunning,
            showModelSelector,
            modelSelectorLoading,
            modelFamilies,
            selectedFamily,
            selectedModel,
            modelOptions,
            isSidebarOpen,
            selectedLabelFilter,
            onFilterChange,
            boundBoxRecalculating,
            labelRegenerating,
            modalImage, boxCanvas, imageResolution,
            openEventModal, onImageLoad, drawBoxes, labelEvent, deleteEvent, retryGemini,

            triggerDailyRoutine, refreshModels, exportDataset, showLogs, fetchLogs,
            regenerateLabels,
            recalculateBoundBoxes,
            toggleModelSelector,
            closeModelSelector,
            reloadModelSelector,
            onFamilyChanged,
            applyModelSelection,
            formatDate, formatEventTime, getStatusClass, getStatusLabel, getLabelClass, getLabelIcon, getConfidenceClass, getFrigateConfidenceClass,
            prevPage, nextPage
        };
    }
}).mount('#app');

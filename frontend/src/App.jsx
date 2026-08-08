import { useState, useRef } from 'react';

function App() {
  const [isLoginMode, setIsLoginMode] = useState(true);
  const [email, setEmail] = useState('admin@daizen.com');
  const [password, setPassword] = useState('password123');
  const [token, setToken] = useState(null);
  const [message, setMessage] = useState('');

  const [projects, setProjects] = useState([]);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');

  const [selectedProject, setSelectedProject] = useState(null);

  const [datasets, setDatasets] = useState([]);
  const [newDatasetName, setNewDatasetName] = useState('');

  const [selectedDataset, setSelectedDataset] = useState(null);
  const [bearings, setBearings] = useState([]);
  const [newBearingName, setNewBearingName] = useState('');

  const [selectedBearing, setSelectedBearing] = useState(null);
  const [dataFiles, setDataFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const [selectedTaskId, setSelectedTaskId] = useState(null);
  const [openedTaskId, setOpenedTaskId] = useState(null);
  const [aiTasks, setAiTasks] = useState([]);
  const [selectedDataFile, setSelectedDataFile] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('백엔드와 통신 중...');
    try {
      let response;
      if (isLoginMode) {
        response = await fetch('http://localhost:8000/api/v1/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          body: JSON.stringify({ email, username: email, password }),
        });
      } else {
        response = await fetch('http://localhost:8000/api/v1/users/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, username: email, password, full_name: "시스템 관리자" }),
        });
      }
      const data = await response.json();
      if (response.ok) {
        if (isLoginMode) {
          const accessToken = data.access_token || data.token;
          setToken(accessToken);
          setMessage('');
          fetchProjects(accessToken);
        } else {
          setMessage('회원가입 성공! 이제 로그인해주세요.');
          setIsLoginMode(true);
        }
      } else {
        setToken(null);
        setMessage(`접근 거부: ${data.detail}`);
      }
    } catch (err) {
      setMessage(`통신 에러: ${err.message}`);
    }
  };

  const fetchProjects = async (authToken) => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/projects/', {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${authToken}`, 'Accept': 'application/json' }
      });
      if (response.ok) {
        const projectData = await response.json();
        setProjects(projectData);
      }
    } catch (err) { }
  };

  const handleCreateProject = async (e) => {
    e.preventDefault();
    if (!newProjectName.trim()) return;
    try {
      const response = await fetch('http://localhost:8000/api/v1/projects/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ name: newProjectName, description: newProjectDescription })
      });
      if (response.ok) {
        setNewProjectName('');
        setNewProjectDescription('');
        fetchProjects(token);
      }
    } catch (err) { }
  };

  const handleDeleteProject = async (projectId, e) => {
    e.stopPropagation();
    if (!window.confirm('정말 이 프로젝트를 삭제하시겠습니까?')) return;
    try {
      const response = await fetch(`http://localhost:8000/api/v1/projects/${projectId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok || response.status === 204) {
        fetchProjects(token);
      }
    } catch (err) { }
  };

  const fetchDatasets = async (projectId) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/datasets/project/${projectId}`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${token}`, 'Accept': 'application/json' }
      });
      if (response.ok) {
        const datasetData = await response.json();
        setDatasets(datasetData);
      } else {
        setDatasets([]);
      }
    } catch (err) {
      setDatasets([]);
    }
  };

  const handleCreateDataset = async (e) => {
    e.preventDefault();
    if (!newDatasetName.trim() || !selectedProject) return;
    try {
      const response = await fetch(`http://localhost:8000/api/v1/datasets/project/${selectedProject.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ name: newDatasetName, description: "프론트엔드 생성" })
      });
      if (response.ok) {
        setNewDatasetName('');
        fetchDatasets(selectedProject.id);
      }
    } catch (err) { }
  };

  const fetchBearings = async (datasetId) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/bearings/dataset/${datasetId}`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${token}`, 'Accept': 'application/json' }
      });
      if (response.ok) {
        const bearingData = await response.json();
        setBearings(bearingData);
      } else {
        setBearings([]);
      }
    } catch (err) {
      setBearings([]);
    }
  };

  const handleCreateBearing = async (e) => {
    e.preventDefault();
    if (!newBearingName.trim() || !selectedDataset) return;
    try {
      const response = await fetch(`http://localhost:8000/api/v1/bearings/dataset/${selectedDataset.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ name: newBearingName, description: "측정 대상" })
      });
      if (response.ok) {
        setNewBearingName('');
        fetchBearings(selectedDataset.id);
      }
    } catch (err) { }
  };

  const fetchDataFiles = async (bearingId) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/data-files/bearing/${bearingId}`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${token}`, 'Accept': 'application/json' }
      });
      if (response.ok) {
        const filesData = await response.json();
        setDataFiles(filesData);
      } else {
        setDataFiles([]);
      }
    } catch (err) {
      setDataFiles([]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile || !selectedBearing) return;
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`http://localhost:8000/api/v1/data-files/bearing/${selectedBearing.id}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });

      if (response.ok) {
        setSelectedFile(null);
        fetchDataFiles(selectedBearing.id);
      } else {
        const data = await response.json();
        alert(`업로드 실패: ${JSON.stringify(data.detail)}`);
      }
    } catch (err) {
      alert(`통신 에러: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const fetchAITasks = async (bearingId) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/ai-tasks/bearing/${bearingId}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setAiTasks(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // 🌟 1. 데이터 파일 삭제 통신 함수 
  const handleDeleteFile = async (fileId, e) => {
    e.stopPropagation();
    if (!window.confirm('정말 이 데이터 파일을 삭제하시겠습니까?')) return;
    try {
      const response = await fetch(`http://localhost:8000/api/v1/data-files/${fileId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok || response.status === 204) {
        fetchDataFiles(selectedBearing.id);
      } else {
        const data = await response.json();
        alert(`삭제 실패: ${data.detail}`);
      }
    } catch (err) {
      alert(`통신 에러: ${err.message}`);
    }
  };

  // 🌟 2. AI 분석 실행 시 필수 파라미터(task_name) 추가
  const handleStartAITask = async () => {
    if (!selectedBearing || !selectedDataFile) {
      alert("분석할 파일을 선택하세요.");
      return;
    }
    setIsAnalyzing(true);
    setAiResult(null);

    try {
      const response = await fetch(`http://localhost:8000/api/v1/ai-tasks/bearing/${selectedBearing.id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          task_name: "predictive_maintenance",
          data_file_id: selectedDataFile.id
        }) // 파라미터 삽입 완료!
      });

      if (response.ok) {
        const task = await response.json();
        pollAITask(task.id);
      } else {
        const errData = await response.json();
        alert(`AI 분석 실패: ${JSON.stringify(errData)}`);
      }
    } catch (err) {
      alert(`통신 에러: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const pollAITask = async (taskId) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          `http://localhost:8000/api/v1/ai-tasks/${taskId}`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (!response.ok) return;

        const task = await response.json();

        if (task.status === "COMPLETED") {
          clearInterval(interval);

          setIsAnalyzing(false);

          await fetchAITasks(selectedBearing.id);

          setAiResult(task);
          setOpenedTaskId(task.id);
        }

        if (task.status === "FAILED") {
          clearInterval(interval);
          setIsAnalyzing(false);
          alert(task.error_message);
        }
      } catch {
        clearInterval(interval);
        setIsAnalyzing(false);
      }
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-10">
      <div className="bg-white p-10 rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-gray-100 max-w-md w-full">

        <h1 className="text-2xl font-bold text-gray-900 mb-2 text-center tracking-tight">DAIZEN AI System</h1>

        <p className="text-sm text-gray-500 mb-6 text-center font-medium">
          {!token ? (isLoginMode ? '관리자 계정으로 로그인' : '새 계정 생성')
            : selectedBearing ? `${selectedDataset?.name} > ${selectedBearing.name}`
              : selectedDataset ? `${selectedProject?.name} > ${selectedDataset.name}`
                : selectedProject ? `${selectedProject.name} > 데이터셋 관리`
                  : '대시보드 메인 화면'}
        </p>

        {!token ? (
          <>
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="border border-gray-200 rounded-xl p-3 focus:outline-none focus:ring-2 focus:ring-black" placeholder="이메일" required />
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="border border-gray-200 rounded-xl p-3 focus:outline-none focus:ring-2 focus:ring-black" placeholder="비밀번호" required />
              <button type="submit" className="bg-black hover:bg-gray-800 text-white font-semibold py-3 px-8 rounded-xl transition-all duration-200 mt-2">
                {isLoginMode ? '로그인' : '회원가입'}
              </button>
            </form>
            <div className="mt-6 text-center">
              <button onClick={() => { setIsLoginMode(!isLoginMode); setMessage(''); }} className="text-sm text-gray-500 hover:text-black transition-colors font-medium underline underline-offset-4">
                {isLoginMode ? '계정이 없으신가요? 회원가입' : '이미 계정이 있으신가요? 로그인'}
              </button>
            </div>
          </>
        )

          /* --- 4단계: 선택된 베어링 내부 (파일 업로드 & AI 분석) --- */
          : selectedBearing ? (
            <div className="flex flex-col gap-6 animate-fade-in">
              <div className="p-5 border border-gray-200 rounded-xl bg-gray-50 flex flex-col gap-4">

                <div className="bg-white p-5 rounded-lg border-2 border-dashed border-gray-300 flex flex-col items-center justify-center gap-3">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden"
                  />

                  {selectedFile ? (
                    <div className="text-center w-full">
                      <p className="text-sm font-medium text-gray-800 mb-3 truncate">📄 {selectedFile.name}</p>
                      <button
                        onClick={handleFileUpload}
                        disabled={isUploading}
                        className={`w-full text-white text-sm font-semibold py-2 px-4 rounded-lg transition-all ${isUploading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
                      >
                        {isUploading ? '업로드 중...' : '서버로 전송'}
                      </button>
                    </div>
                  ) : (
                    <div className="text-center">
                      <p className="text-sm text-gray-500 mb-3">분석할 데이터 파일을 업로드하세요.</p>
                      <button onClick={() => fileInputRef.current.click()} className="bg-gray-800 hover:bg-black text-white text-sm font-semibold py-2 px-4 rounded-lg transition-all">
                        파일 선택하기
                      </button>
                    </div>
                  )}
                </div>

                <div>
                  <h3 className="font-bold text-gray-800 mb-2 text-sm">업로드된 파일 목록</h3>
                  {dataFiles.length === 0 ? (
                    <p className="text-sm text-gray-400 text-center py-4 bg-white border border-gray-200 rounded-lg">파일이 없습니다.</p>
                  ) : (
                    <div className="flex flex-col gap-4">
                      <ul className="flex flex-col gap-2">
                        {dataFiles.map((f, idx) => (
                          <li
                            key={idx}
                            onClick={() => setSelectedDataFile(f)}
                            className={`p-3 rounded-lg border text-sm font-medium shadow-sm flex justify-between items-center group cursor-pointer hover:border-black transition-all ${selectedDataFile?.id === f.id
                              ? "border-black bg-gray-50"
                              : "border-gray-200 bg-white"
                              }`}
                          >
                            <div className="flex items-center gap-2 overflow-hidden">
                              <span className="text-gray-900 truncate">📄 {f.original_filename || f.file_name || `파일 #${idx + 1}`}</span>
                              <span className="text-xs text-gray-400">{f.file_size ? `${(f.file_size / 1024).toFixed(1)}KB` : ''}</span>
                              {selectedDataFile?.id === f.id && (
                                <div className="text-xs text-green-600 font-semibold">
                                  ✓ 분석 대상
                                </div>
                              )}
                            </div>
                            {/* 🌟 3. 파일 삭제 버튼 적용 */}
                            <button
                              onClick={(e) => handleDeleteFile(f.id, e)}
                              className="text-red-500 hover:text-red-700 hover:bg-red-50 px-2 py-1 rounded transition-colors text-xs font-semibold opacity-0 group-hover:opacity-100"
                            >
                              삭제
                            </button>
                          </li>
                        ))}
                      </ul>

                      <div className="bg-blue-50 p-5 rounded-lg border border-blue-100 flex flex-col items-center justify-center gap-3">
                        <h3 className="font-bold text-blue-900 text-sm">🧠 데이터 분석 및 예측</h3>
                        <p className="text-xs text-blue-700 text-center mb-1">
                          업로드된 데이터를 바탕으로 AI 모델을 구동하여 상태를 진단합니다.
                        </p>
                        <button
                          onClick={handleStartAITask}
                          disabled={isAnalyzing}
                          className={`w-full text-white text-sm font-bold py-3 px-4 rounded-xl transition-all shadow-md ${isAnalyzing
                            ? 'bg-blue-300 cursor-not-allowed'
                            : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 transform hover:-translate-y-0.5'
                            }`}
                        >
                          {isAnalyzing ? 'AI가 데이터를 분석 중입니다...' : '🚀 AI 분석 시작하기'}
                        </button>


                        {aiTasks.length > 0 && (
                          <div className="mt-6">
                            <h3 className="font-bold text-gray-800 mb-2">📋 분석 이력</h3>

                            <div className="space-y-2">
                              {[...aiTasks]
                                .sort((a, b) => b.id - a.id)
                                .slice(0, 3)
                                .map((task) => (
                                  <div key={task.id}>
                                    <button
                                      onClick={() => {
                                        setAiResult(task);
                                        setOpenedTaskId(
                                          openedTaskId === task.id ? null : task.id
                                        );
                                      }}
                                      className="w-full p-3 rounded-lg border border-gray-200 bg-white text-left hover:border-blue-500"
                                    >
                                      <div className="flex justify-between items-center">
                                        <div>
                                          <div className="font-medium">
                                            📄 {task.file_name}
                                          </div>

                                          <div className="text-xs text-gray-500">
                                            {task.created_at
                                              ? new Date(task.created_at).toLocaleString()
                                              : ""}
                                          </div>
                                        </div>

                                        <span
                                          className={`text-xs font-semibold ${task.status === "COMPLETED"
                                            ? "text-green-600"
                                            : task.status === "RUNNING"
                                              ? "text-blue-600"
                                              : "text-gray-500"
                                            }`}
                                        >
                                          {task.status}
                                        </span>
                                      </div>
                                    </button>

                                    {openedTaskId === task.id && aiResult?.id === task.id && (
                                      <div className="mt-2 mb-3 p-4 bg-white rounded-lg border border-blue-200">
                                        <h4 className="font-bold mb-2">
                                          📊 분석 결과
                                        </h4>

                                        <pre className="text-xs whitespace-pre-wrap overflow-auto max-h-72">
                                          {aiResult.result_data
                                            ? JSON.parse(aiResult.result_data).huggingface_result
                                              .choices[0].message.content
                                            : "AI 분석 결과가 없습니다."}
                                        </pre>
                                      </div>
                                    )}
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

              </div>

              <button
                onClick={() => { setSelectedBearing(null); setSelectedFile(null); setAiResult(null); }}
                className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-3 px-4 rounded-xl transition-all text-sm"
              >
                ← 이전 화면으로 돌아가기
              </button>
            </div>
          )

            /* --- 3단계, 2단계, 1단계 화면 (변경 없음) --- */
            : selectedDataset ? (
              <div className="flex flex-col gap-6 animate-fade-in">
                <div className="p-5 border border-gray-200 rounded-xl bg-gray-50 flex flex-col gap-4">
                  <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                    <h3 className="font-bold text-gray-800 mb-2 text-sm">⚙️ 측정 대상(베어링) 추가</h3>
                    <form onSubmit={handleCreateBearing} className="flex gap-2">
                      <input type="text" value={newBearingName} onChange={(e) => setNewBearingName(e.target.value)} className="border border-gray-200 rounded-lg p-2 text-sm flex-1 focus:outline-none focus:ring-2 focus:ring-black" placeholder="예: 1번 설비 스핀들" required />
                      <button type="submit" className="bg-gray-800 hover:bg-black text-white text-sm font-semibold py-2 px-4 rounded-lg transition-all">추가</button>
                    </form>
                  </div>
                  <div>
                    <h3 className="font-bold text-gray-800 mb-2 text-sm">등록된 측정 대상</h3>
                    {bearings.length === 0 ? (
                      <p className="text-sm text-gray-400 text-center py-4 bg-white border border-dashed border-gray-300 rounded-lg">등록된 대상이 없습니다.</p>
                    ) : (
                      <ul className="flex flex-col gap-2">
                        {bearings.map((b, idx) => (
                          <li key={idx} onClick={() => { setSelectedBearing(b); fetchDataFiles(b.id); fetchAITasks(b.id); }} className="p-3 bg-white rounded-lg border border-gray-200 text-sm font-medium shadow-sm flex justify-between items-center cursor-pointer hover:border-black transition-all group">
                            <span className="text-gray-900 group-hover:text-black">{b.name || `베어링 #${idx + 1}`}</span>
                            <span className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded-md border border-green-200 group-hover:bg-green-100 transition-colors">데이터 업로드</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
                <button onClick={() => { setSelectedDataset(null); setBearings([]); }} className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-3 px-4 rounded-xl transition-all text-sm">
                  ← 데이터셋 목록으로 돌아가기
                </button>
              </div>
            ) : selectedProject ? (
              <div className="flex flex-col gap-6 animate-fade-in">
                <div className="p-5 border border-gray-200 rounded-xl bg-gray-50 flex flex-col gap-4">
                  <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                    <h3 className="font-bold text-gray-800 mb-2 text-sm">📁 새 데이터셋 그룹 만들기</h3>
                    <form onSubmit={handleCreateDataset} className="flex gap-2">
                      <input type="text" value={newDatasetName} onChange={(e) => setNewDatasetName(e.target.value)} className="border border-gray-200 rounded-lg p-2 text-sm flex-1 focus:outline-none focus:ring-2 focus:ring-black" placeholder="예: 2026년 상반기 측정 데이터" required />
                      <button type="submit" className="bg-gray-800 hover:bg-black text-white text-sm font-semibold py-2 px-4 rounded-lg transition-all">추가</button>
                    </form>
                  </div>
                  <div>
                    <h3 className="font-bold text-gray-800 mb-2 text-sm">현재 프로젝트의 데이터셋</h3>
                    {datasets.length === 0 ? (
                      <p className="text-sm text-gray-400 text-center py-4 bg-white border border-dashed border-gray-300 rounded-lg">생성된 데이터셋이 없습니다.</p>
                    ) : (
                      <ul className="flex flex-col gap-2">
                        {datasets.map((ds, idx) => (
                          <li key={idx} onClick={() => { setSelectedDataset(ds); fetchBearings(ds.id); }} className="p-3 bg-white rounded-lg border border-gray-200 text-sm font-medium shadow-sm flex justify-between items-center cursor-pointer hover:border-black transition-all group">
                            <span className="text-gray-900 group-hover:text-black">{ds.name || `데이터셋 #${idx + 1}`}</span>
                            <span className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded-md group-hover:bg-blue-100 transition-colors">열기 →</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
                <button onClick={() => { setSelectedProject(null); setDatasets([]); }} className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-3 px-4 rounded-xl transition-all text-sm">
                  ← 프로젝트 목록으로 돌아가기
                </button>
              </div>
            ) : (
              <div className="flex flex-col gap-6">
                <div className="p-4 border border-gray-200 rounded-xl bg-gray-50">
                  <h3 className="font-bold text-gray-800 mb-3 text-sm">✨ 새 프로젝트 만들기</h3>
                  <form onSubmit={handleCreateProject} className="flex flex-col gap-2">
                    <input type="text" value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} className="border border-gray-200 rounded-lg p-2 text-sm focus:outline-none focus:ring-2 focus:ring-black" placeholder="프로젝트 이름" required />
                    <input type="text" value={newProjectDescription} onChange={(e) => setNewProjectDescription(e.target.value)} className="border border-gray-200 rounded-lg p-2 text-sm focus:outline-none focus:ring-2 focus:ring-black" placeholder="프로젝트 설명 (선택)" />
                    <button type="submit" className="bg-gray-800 hover:bg-black text-white font-semibold py-2 px-4 rounded-lg transition-all text-sm mt-1">생성</button>
                  </form>
                </div>
                <div>
                  <h3 className="font-bold text-gray-800 mb-2">진행 중인 프로젝트</h3>
                  {projects.length === 0 ? (
                    <p className="text-sm text-gray-400">등록된 프로젝트가 없습니다.</p>
                  ) : (
                    <ul className="flex flex-col gap-3">
                      {projects.map((p, idx) => (
                        <li key={idx} onClick={() => { setSelectedProject(p); fetchDatasets(p.id); }} className="p-4 bg-white rounded-xl border border-gray-200 text-sm font-medium shadow-sm flex flex-col gap-1 cursor-pointer hover:border-black hover:shadow-md transition-all group">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-900 font-bold group-hover:text-black">{p.name}</span>
                            <span className="text-gray-400 text-xs">입장하기 →</span>
                          </div>
                          <div className="flex justify-between items-end mt-1">
                            <span className="text-gray-500 text-xs">{p.description}</span>
                            <button onClick={(e) => handleDeleteProject(p.id, e)} className="text-red-500 hover:text-red-700 hover:bg-red-50 px-2 py-1 rounded transition-colors text-xs font-semibold">삭제</button>
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <button onClick={() => { setToken(null); setProjects([]); }} className="text-sm text-gray-400 hover:text-gray-600 underline underline-offset-4 mt-2">로그아웃</button>
              </div>
            )}

        {message && (
          <div className="mt-6 p-4 bg-red-50 rounded-xl border border-red-100 text-sm text-center text-red-600 font-medium break-words">
            {message}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
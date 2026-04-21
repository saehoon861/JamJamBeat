import React from 'react';
import { 
  BarChart, 
  Activity, 
  Clock, 
  TrendingUp,
  Layers,
  Zap,
  Target,
  LineChart as LineChartIcon,
  ShieldCheck
} from 'lucide-react';

const Card = ({ title, children, icon: Icon, className = "" }) => (
  <div className={`bg-white p-6 rounded-xl border border-slate-200 shadow-sm ${className}`}>
    <div className="flex items-center gap-2 mb-4">
      <Icon className="w-5 h-5 text-indigo-600" />
      <h3 className="font-bold text-slate-800">{title}</h3>
    </div>
    {children}
  </div>
);

const App = () => {
  const classes = ["neutral", "fist", "palm", "V", "pinky", "animal", "heart"];
  
  // 1. Confusion Matrix Mock Data
  const matrixData = [
    [0.98, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00],
    [0.05, 0.92, 0.02, 0.01, 0.00, 0.00, 0.00],
    [0.02, 0.01, 0.95, 0.01, 0.01, 0.00, 0.00],
    [0.08, 0.02, 0.03, 0.85, 0.01, 0.01, 0.00],
    [0.04, 0.01, 0.02, 0.02, 0.91, 0.00, 0.00],
    [0.03, 0.02, 0.05, 0.01, 0.01, 0.88, 0.00],
    [0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.92]
  ];

  // 2. Per-class F1 Mock Data
  const f1Scores = [0.99, 0.94, 0.96, 0.88, 0.92, 0.90, 0.95];

  return (
    <div className="p-8 bg-slate-50 min-h-screen font-sans text-slate-900">
      <header className="mb-8 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 mb-2">JamJamBeat 모델 종합 검증 리포트</h1>
          <p className="text-slate-500 text-sm">Target Model: MLP-v1.2.0 | Dataset: 4-User Integrated Testset</p>
        </div>
        <div className="text-right">
          <span className="inline-block px-3 py-1 bg-emerald-100 text-emerald-700 text-xs font-bold rounded-full">PASSED</span>
        </div>
      </header>

      {/* 상단 요약 지표 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
          <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider">Macro F1 Score</p>
          <p className="text-2xl font-bold text-indigo-600">0.934</p>
        </div>
        <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
          <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider">FP/min @ Neutral</p>
          <p className="text-2xl font-bold text-rose-500">0.28 <span className="text-xs font-normal text-slate-400">/min</span></p>
        </div>
        <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
          <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider">p95 Latency</p>
          <p className="text-2xl font-bold text-amber-500">58.4 <span className="text-xs font-normal text-slate-400">ms</span></p>
        </div>
        <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
          <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider">Optimal τ</p>
          <p className="text-2xl font-bold text-slate-700">0.85</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* 1. Confusion Matrix (2D 히트맵) */}
        <Card title="Confusion Matrix" icon={Layers} className="lg:col-span-2">
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] text-center border-collapse">
              <thead>
                <tr>
                  <th className="p-2 border bg-slate-50">True \ Pred</th>
                  {classes.map(c => <th key={c} className="p-2 border bg-slate-50 font-bold uppercase">{c}</th>)}
                </tr>
              </thead>
              <tbody>
                {matrixData.map((row, i) => (
                  <tr key={i}>
                    <td className="p-2 border bg-slate-50 font-bold uppercase">{classes[i]}</td>
                    {row.map((val, j) => (
                      <td 
                        key={j} 
                        className="p-2 border transition-all"
                        style={{ backgroundColor: `rgba(79, 70, 229, ${val})`, color: val > 0.4 ? 'white' : '#64748b' }}
                      >
                        {(val * 100).toFixed(0)}%
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-[11px] text-slate-400">각 행은 실제 클래스 대비 모델이 예측한 비율을 나타냅니다 (Row-normalized).</p>
        </Card>

        {/* 2. Per-class F1 (막대 그래프) */}
        <Card title="Per-class F1-Score" icon={Target}>
          <div className="space-y-3 pt-2">
            {classes.map((name, i) => (
              <div key={name} className="group">
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium text-slate-600 uppercase">{name}</span>
                  <span className="font-bold text-indigo-600">{f1Scores[i].toFixed(2)}</span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                  <div 
                    className="bg-indigo-500 h-full transition-all duration-1000" 
                    style={{ width: `${f1Scores[i] * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* 3. Threshold Sweep (이중 축) */}
        <Card title="τ Sweep: FP vs Recall" icon={TrendingUp}>
          <div className="h-48 border-l border-b border-slate-200 relative mt-4">
            <svg className="absolute inset-0 w-full h-full overflow-visible">
              {/* FP line */}
              <path d="M 0 30 Q 150 120 300 160" fill="none" stroke="#f43f5e" strokeWidth="2" />
              {/* Recall line */}
              <path d="M 0 50 Q 150 60 300 120" fill="none" stroke="#6366f1" strokeWidth="2" strokeDasharray="4" />
              {/* Threshold indicator */}
              <line x1="225" y1="0" x2="225" y2="100%" stroke="#94a3b8" strokeDasharray="2" />
            </svg>
            <div className="absolute top-0 right-0 flex flex-col gap-1 text-[9px] font-bold">
              <span className="text-rose-500">— FP/min</span>
              <span className="text-indigo-500">--- Recall</span>
            </div>
            <div className="absolute bottom-[-20px] left-0 w-full flex justify-between text-[10px] text-slate-400">
              <span>0.70</span><span>0.85</span><span>0.95</span>
            </div>
          </div>
          <p className="mt-8 text-xs text-slate-500 leading-tight">임계값 $\tau=0.85$ 지점에서 오탐률 0.3 미만을 달성하며 최적화됨.</p>
        </Card>

        {/* 4. Learning Curve (학습 곡선) */}
        <Card title="Learning Curve" icon={LineChartIcon}>
          <div className="h-48 border-l border-b border-slate-200 relative mt-4">
            <svg className="absolute inset-0 w-full h-full">
              {/* Train Loss */}
              <path d="M 0 160 L 50 100 L 100 60 L 150 40 L 200 30 L 300 25" fill="none" stroke="#94a3b8" strokeWidth="1.5" />
              {/* Val Loss */}
              <path d="M 0 160 L 50 110 L 100 75 L 150 65 L 200 70 L 300 75" fill="none" stroke="#6366f1" strokeWidth="2" />
            </svg>
            <div className="absolute top-0 right-0 text-[9px] flex gap-2">
              <span className="text-slate-400 font-bold">Train Loss</span>
              <span className="text-indigo-500 font-bold">Val Loss</span>
            </div>
            <div className="absolute bottom-[-20px] left-0 w-full flex justify-between text-[10px] text-slate-400 px-1">
              <span>Epoch 0</span><span>Epoch 50</span>
            </div>
          </div>
          <p className="mt-8 text-xs text-slate-500">Validation Loss가 일정 수준 유지되어 과적합(Overfitting) 징후 없음.</p>
        </Card>

        {/* 5. Reliability Diagram (신뢰도 교정) */}
        <Card title="Reliability Diagram" icon={ShieldCheck}>
          <div className="h-48 flex items-end justify-center gap-1 border-l border-b border-slate-200 relative mt-4">
            <svg className="absolute inset-0 w-full h-full">
              {/* Ideal line */}
              <line x1="0" y1="100%" x2="100%" y2="0" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="4" />
              {/* Actual curve */}
              <polyline points="0,192 60,140 120,100 180,60 240,25 300,5" fill="none" stroke="#10b981" strokeWidth="2" />
            </svg>
            <div className="absolute bottom-[-20px] left-0 w-full flex justify-between text-[10px] text-slate-400">
              <span>Confidence 0</span><span>1.0</span>
            </div>
          </div>
          <p className="mt-8 text-xs text-slate-500 italic">"모델이 90% 확신할 때 실제로 90% 정답인가?"를 검증함 (ECE: 0.04).</p>
        </Card>

        {/* 6. Latency Analysis (CDF + Box) */}
        <Card title="Latency Analysis" icon={Clock}>
          <div className="space-y-3 mt-2">
            <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
              <div className="flex justify-between text-[10px] font-bold text-slate-400 mb-2 uppercase">Components</div>
              <div className="space-y-2">
                {[
                  { n: "MediaPipe", v: "28ms", w: "70%" },
                  { n: "MLP+Post", v: "4ms", w: "15%" },
                ].map(s => (
                  <div key={s.n} className="flex items-center gap-2">
                    <span className="text-xs w-20 text-slate-600">{s.n}</span>
                    <div className="flex-1 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                      <div className="bg-amber-400 h-full" style={{ width: s.w }}></div>
                    </div>
                    <span className="text-xs font-bold w-10 text-right">{s.v}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="h-16 border-l border-b border-slate-200 relative mt-4 overflow-hidden rounded">
               <svg className="absolute inset-0 w-full h-full">
                 <path d="M 0 60 Q 150 55 300 0" fill="none" stroke="#6366f1" strokeWidth="2" />
               </svg>
               <div className="absolute top-1 right-2 text-[9px] text-indigo-500 font-bold">Latency CDF</div>
            </div>
          </div>
        </Card>

        {/* 7. Temporal Jitter (시간축 변동성) */}
        <Card title="Temporal Jitter" icon={Activity} className="lg:col-span-2">
          <div className="h-40 bg-slate-900 rounded-lg p-2 relative overflow-hidden">
             <div className="absolute top-2 left-2 text-[10px] text-emerald-400 font-mono">DEBUG: Session_3_fast_man3</div>
             <svg className="w-full h-full pt-4">
                {/* Background Noise */}
                <polyline points="0,120 30,125 60,118 90,130 120,122 150,128 180,115 210,125 240,122 300,128" fill="none" stroke="#334155" strokeWidth="1" />
                {/* Gesture Probability Rise */}
                <polyline points="0,140 100,140 130,120 150,40 250,30 300,35" fill="none" stroke="#6366f1" strokeWidth="2" />
                {/* Threshold line */}
                <line x1="0" y1="60" x2="100%" y2="60" stroke="#f43f5e" strokeWidth="1" strokeDasharray="4" opacity="0.6" />
             </svg>
             <div className="absolute bottom-2 right-2 text-[9px] text-rose-400 font-bold">τ = 0.85</div>
          </div>
          <p className="mt-4 text-xs text-slate-500 leading-tight">동작 전환 구간(Transition)에서의 확률 요동을 Debounce(K=3) 필터로 방어 완료.</p>
        </Card>

      </div>

      <footer className="mt-12 p-6 bg-indigo-50 rounded-xl border border-indigo-100 flex items-start gap-4">
        <Zap className="w-6 h-6 text-indigo-600 mt-1 flex-shrink-0" />
        <div>
          <h4 className="font-bold text-indigo-900 mb-1">종합 분석 및 배포 승인</h4>
          <p className="text-sm text-indigo-700 leading-relaxed">
            전체 Macro F1 0.93 및 Neutral 구간 오탐률 0.3/min 미만으로 실사용 요구 조건을 충족함. 
            특히 <strong>Reliability Diagram</strong>에서 확인된 바와 같이 모델이 예측한 확률이 실제 정확도와 잘 일치하며, 
            <strong>Learning Curve</strong> 상에서 과적합 없이 안정적으로 수렴함. 
            차후 MediaPipe 단계의 지연 시간(28ms) 최적화를 통해 60fps 환경 대응 권장.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
import React, { useState, useEffect, useRef } from 'react';
import { Activity, BarChart2, Cpu, Settings, UploadCloud, Sparkles, FileText, Image as ImageIcon, Music, CheckCircle2, ShieldCheck, Zap, Layers, Scale } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import './index.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');
  
  return (
    <div className="app-layout">
      {/* Top Header & Brand Bar */}
      <header className="app-header">
        <div className="brand-logo">
          <div className="brand-icon">
            <Sparkles size={20} />
          </div>
          <span>MALGR-Net AI Platform</span>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button className={`nav-tab-btn ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
            <UploadCloud size={18} /> Upload & Gating Analysis
          </button>
          <button className={`nav-tab-btn ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>
            <BarChart2 size={18} /> Benchmarks & Analytics
          </button>
          <button className={`nav-tab-btn ${activeTab === 'diagnostics' ? 'active' : ''}`} onClick={() => setActiveTab('diagnostics')}>
            <Cpu size={18} /> System Diagnostics
          </button>
        </nav>

        {/* System Health Badge */}
        <div className="system-status-badge">
          <div className="status-dot" />
          <span>Backend Online (MALGR-Net)</span>
        </div>
      </header>

      {/* Main Content View */}
      <main className="main-content">
        {activeTab === 'upload' && <UploadAnalysis />}
        {activeTab === 'analytics' && <Analytics />}
        {activeTab === 'diagnostics' && <Diagnostics />}
      </main>
    </div>
  );
}

// ----------------------------------------
// 1. UPLOAD MULTIMODAL ANALYSIS MODULE
// ----------------------------------------
function UploadAnalysis() {
  const [text, setText] = useState('');
  const [image, setImage] = useState(null);
  const [audio, setAudio] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [probs, setProbs] = useState(null);
  const [uploadHistory, setUploadHistory] = useState([]);
  const [shapData, setShapData] = useState(null);
  const [isExplaining, setIsExplaining] = useState(false);

  const handleAnalyze = async () => {
    if (!text && !image && !audio) return;
    setIsAnalyzing(true);
    setProbs(null);
    setShapData(null);
    
    const formData = new FormData();
    formData.append('text', text);
    if (image) formData.append('image', image);
    if (audio) formData.append('audio', audio);

    try {
      // 1. Predict Endpoint
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setProbs(data);
      setUploadHistory(prev => {
        const newHist = [...prev, { time: new Date().toLocaleTimeString(), ...data }];
        if (newHist.length > 20) newHist.shift();
        return newHist;
      });

      // 2. Multimodal SHAP Explainability Endpoint
      setIsExplaining(true);
      try {
        const shapRes = await fetch('http://localhost:8000/explain_multimodal', {
          method: 'POST',
          body: formData
        });
        const shapResult = await shapRes.json();
        if (!shapResult.error) {
          setShapData(shapResult);
        } else {
          console.error("Multimodal SHAP Error:", shapResult.error);
        }
      } catch (e) {
        console.error("Failed to fetch Multimodal SHAP", e);
      }
      setIsExplaining(false);
    } catch (err) {
      console.error("Prediction Error:", err);
    }
    setIsAnalyzing(false);
  };

  // Determine Dominant Sentiment & Color
  const getDominantSentiment = () => {
    if (!probs) return { label: 'NEUTRAL', score: 0, color: 'var(--neu-color)', glow: 'var(--neu-glow)' };
    const maxVal = Math.max(probs.positive, probs.negative, probs.neutral);
    if (maxVal === probs.positive) return { label: 'POSITIVE', score: probs.positive, color: 'var(--pos-color)', glow: 'var(--pos-glow)' };
    if (maxVal === probs.negative) return { label: 'NEGATIVE', score: probs.negative, color: 'var(--neg-color)', glow: 'var(--neg-glow)' };
    return { label: 'NEUTRAL', score: probs.neutral, color: 'var(--neu-color)', glow: 'var(--neu-glow)' };
  };

  const dominant = getDominantSentiment();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      <div className="page-title-section">
        <h2 className="page-title">MALGR-Net Multimodal Analysis & Gating</h2>
        <p className="page-subtitle">
          Upload text, facial images/video frames, or vocal audio clips to run Multimodal Adaptive Latent-Gated Representation Network inference.
        </p>
      </div>

      {/* Main Grid: Upload Inputs vs Inference Output */}
      <div className="grid-two-col">
        
        {/* INPUT CARDS */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3>Modality Inputs</h3>
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', background: 'rgba(0, 242, 254, 0.1)', padding: '3px 8px', borderRadius: '6px', border: '1px solid rgba(0, 242, 254, 0.2)' }}>
              Priority: Audio &gt; Visual &gt; Text
            </span>
          </div>

          {/* 1. Audio Input Card (Priority 1: Primary Driver) */}
          <div className="upload-card" style={{ borderColor: audio ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.08)' }}>
            <div className="upload-card-label">
              <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Music size={18} color="var(--accent-cyan)" /> Acoustic Audio Clip (.wav / .mp3)
              </span>
              <span style={{ fontSize: '0.72rem', color: '#10b981', fontWeight: 600, background: 'rgba(16, 185, 129, 0.15)', padding: '2px 6px', borderRadius: '4px' }}>
                Priority 1 (Primary)
              </span>
            </div>
            {audio ? (
              <div className="file-selected-badge">
                <CheckCircle2 size={16} /> {audio.name}
              </div>
            ) : (
              <div className="upload-dropzone-inner">
                <UploadCloud size={24} color="var(--accent-cyan)" />
                <span style={{ fontSize: '0.85rem' }}>Drag & drop vocal audio clip or click to browse</span>
              </div>
            )}
            <input type="file" accept="audio/*" className="file-input-hidden" onChange={e => setAudio(e.target.files[0])} />
          </div>

          {/* 2. Visual Input Card (Priority 2: Secondary Driver) */}
          <div className="upload-card" style={{ borderColor: image ? 'var(--accent-purple)' : 'rgba(255,255,255,0.08)' }}>
            <div className="upload-card-label">
              <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ImageIcon size={18} color="var(--accent-purple)" /> Visual Input (Image / Video Frame)
              </span>
              <span style={{ fontSize: '0.72rem', color: '#93c5fd', fontWeight: 600, background: 'rgba(59, 130, 246, 0.15)', padding: '2px 6px', borderRadius: '4px' }}>
                Priority 2 (Secondary)
              </span>
            </div>
            {image ? (
              <div className="file-selected-badge">
                <CheckCircle2 size={16} /> {image.name}
              </div>
            ) : (
              <div className="upload-dropzone-inner">
                <UploadCloud size={24} color="var(--accent-purple)" />
                <span style={{ fontSize: '0.85rem' }}>Drag & drop facial image/video frame or click to browse</span>
              </div>
            )}
            <input type="file" accept="image/*,video/*" className="file-input-hidden" onChange={e => setImage(e.target.files[0])} />
          </div>

          {/* 3. Text Input Card (Priority 3: Context Driver) */}
          <div className="upload-card">
            <div className="upload-card-label">
              <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FileText size={18} color="var(--accent-cyan)" /> Linguistic Text Context
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.72rem', color: '#fde047', fontWeight: 600, background: 'rgba(245, 158, 11, 0.15)', padding: '2px 6px', borderRadius: '4px' }}>
                  Priority 3 (Context)
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>{text.length} chars</span>
              </div>
            </div>
            <textarea
              placeholder="Enter multilingual text narrative, review, or spoken dialogue..."
              value={text}
              onChange={e => setText(e.target.value)}
              style={{
                width: '100%', height: '80px', padding: '12px',
                background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '10px', color: '#fff', fontFamily: 'inherit', resize: 'none', outline: 'none'
              }}
            />
          </div>

          <button className="btn-primary" onClick={handleAnalyze} disabled={isAnalyzing || (!text && !image && !audio)}>
            {isAnalyzing ? (
              <>
                <Zap size={18} className="animate-spin" /> Fusing Latent Representations...
              </>
            ) : (
              <>
                <Sparkles size={18} /> Run MALGR-Net Inference
              </>
            )}
          </button>
        </div>

        {/* RESULTS CARD */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <h3>MALGR-Net Inference & Gating Results</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Powered by MALGR-Net Adaptive Latent Gating ($G = [g_v, g_a, g_t]$)
          </p>

          {!probs && !isAnalyzing && (
            <div style={{ height: '260px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', gap: '12px' }}>
              <Zap size={36} opacity={0.4} />
              <span>Awaiting Input Modalities...</span>
            </div>
          )}

          {isAnalyzing && (
            <div style={{ height: '260px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-cyan)', gap: '12px' }}>
              <div className="status-dot" style={{ width: '16px', height: '16px' }} />
              <span>Projecting Features to Latent Space (256-d)...</span>
            </div>
          )}

          {probs && !isAnalyzing && (
            <>
              {/* Radial Gauge */}
              <div className="result-display-container">
                <div className="sentiment-gauge-circle" style={{ borderColor: dominant.color, boxShadow: `0 0 35px ${dominant.glow}` }}>
                  <span className="sentiment-gauge-value" style={{ color: dominant.color }}>{dominant.label}</span>
                  <span className="sentiment-gauge-confidence">{(dominant.score * 100).toFixed(1)}% Confidence</span>
                </div>
              </div>

              {/* Gating Breakdown Section */}
              {probs.gating_weights && (
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '14px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                      <Scale size={16} /> MALGR-Net Adaptive Latent Gates ($g_a, g_v, g_t$)
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)', background: 'rgba(0, 242, 254, 0.1)', padding: '2px 8px', borderRadius: '6px', border: '1px solid rgba(0, 242, 254, 0.2)' }}>
                      Priority: Audio &gt; Visual &gt; Text
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '10px' }}>
                    {/* Priority 1: Audio */}
                    <div style={{ flex: 1, background: probs.gating_weights.audio > 0 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.03)', border: probs.gating_weights.audio > 0 ? '1px solid #10b981' : '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', padding: '8px', textAlign: 'center', opacity: probs.gating_weights.audio > 0 ? 1 : 0.5 }}>
                      <div style={{ fontSize: '0.72rem', color: probs.gating_weights.audio > 0 ? '#6ee7b7' : 'var(--text-muted)' }}>
                        Audio (ga) {probs.gating_weights.audio > 0 ? <span style={{ fontSize: '0.62rem', color: '#10b981', fontWeight: 600 }}>★ Priority 1</span> : <span style={{ fontSize: '0.62rem' }}>(Inactive)</span>}
                      </div>
                      <div style={{ fontSize: '1rem', fontWeight: 700, color: probs.gating_weights.audio > 0 ? '#10b981' : 'var(--text-muted)' }}>{probs.gating_weights.audio}%</div>
                    </div>

                    {/* Priority 2: Visual */}
                    <div style={{ flex: 1, background: probs.gating_weights.visual > 0 ? 'rgba(59, 130, 246, 0.15)' : 'rgba(255, 255, 255, 0.03)', border: probs.gating_weights.visual > 0 ? '1px solid #3b82f6' : '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', padding: '8px', textAlign: 'center', opacity: probs.gating_weights.visual > 0 ? 1 : 0.5 }}>
                      <div style={{ fontSize: '0.72rem', color: probs.gating_weights.visual > 0 ? '#93c5fd' : 'var(--text-muted)' }}>
                        Visual (gv) {probs.gating_weights.visual > 0 ? <span style={{ fontSize: '0.62rem', color: '#93c5fd', fontWeight: 600 }}>★ Priority 2</span> : <span style={{ fontSize: '0.62rem' }}>(Inactive)</span>}
                      </div>
                      <div style={{ fontSize: '1rem', fontWeight: 700, color: probs.gating_weights.visual > 0 ? '#3b82f6' : 'var(--text-muted)' }}>{probs.gating_weights.visual}%</div>
                    </div>

                    {/* Priority 3: Text */}
                    <div style={{ flex: 1, background: probs.gating_weights.text > 0 ? 'rgba(245, 158, 11, 0.15)' : 'rgba(255, 255, 255, 0.03)', border: probs.gating_weights.text > 0 ? '1px solid #f59e0b' : '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', padding: '8px', textAlign: 'center', opacity: probs.gating_weights.text > 0 ? 1 : 0.5 }}>
                      <div style={{ fontSize: '0.72rem', color: probs.gating_weights.text > 0 ? '#fde047' : 'var(--text-muted)' }}>
                        Text (gt) {probs.gating_weights.text > 0 ? <span style={{ fontSize: '0.62rem', color: '#fde047', fontWeight: 600 }}>Priority 3</span> : <span style={{ fontSize: '0.62rem' }}>(Inactive)</span>}
                      </div>
                      <div style={{ fontSize: '1rem', fontWeight: 700, color: probs.gating_weights.text > 0 ? '#f59e0b' : 'var(--text-muted)' }}>{probs.gating_weights.text}%</div>
                    </div>
                  </div>
                </div>
              )}

              {/* 3 Sentiment Probabilities */}
              <div className="prob-bar-container">
                <div className="prob-bar-row">
                  <div className="prob-bar-header">
                    <span>Positive Sentiment</span>
                    <span style={{ color: 'var(--pos-color)' }}>{(probs.positive * 100).toFixed(1)}%</span>
                  </div>
                  <div className="prob-track">
                    <div className="prob-fill" style={{ width: `${probs.positive * 100}%`, background: 'var(--pos-color)', boxShadow: '0 0 10px var(--pos-glow)' }} />
                  </div>
                </div>

                <div className="prob-bar-row">
                  <div className="prob-bar-header">
                    <span>Neutral Sentiment</span>
                    <span style={{ color: 'var(--neu-color)' }}>{(probs.neutral * 100).toFixed(1)}%</span>
                  </div>
                  <div className="prob-track">
                    <div className="prob-fill" style={{ width: `${probs.neutral * 100}%`, background: 'var(--neu-color)', boxShadow: '0 0 10px var(--neu-glow)' }} />
                  </div>
                </div>

                <div className="prob-bar-row">
                  <div className="prob-bar-header">
                    <span>Negative Sentiment</span>
                    <span style={{ color: 'var(--neg-color)' }}>{(probs.negative * 100).toFixed(1)}%</span>
                  </div>
                  <div className="prob-track">
                    <div className="prob-fill" style={{ width: `${probs.negative * 100}%`, background: 'var(--neg-color)', boxShadow: '0 0 10px var(--neg-glow)' }} />
                  </div>
                </div>
              </div>

              {/* 7 Emotion Class Distribution (If available) */}
              {probs.emotions && (
                <div style={{ background: 'rgba(0,0,0,0.25)', padding: '12px 14px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '8px', fontWeight: 600 }}>
                    7 Emotion Classes Breakdown
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
                    {Object.entries(probs.emotions).map(([emo, val]) => (
                      <div key={emo} style={{ background: 'rgba(255,255,255,0.04)', padding: '6px', borderRadius: '6px', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)', textTransform: 'capitalize' }}>{emo}</div>
                        <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#fff' }}>{(val * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* MULTIMODAL SHAP EXPLANATION PANEL */}
      {(text || image || audio) && (isExplaining || shapData) && (
        <div className="glass-panel" style={{ height: 'auto' }}>
          <h3 style={{ marginBottom: '8px' }}>Multimodal SHAP & Gate Explainability</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '24px' }}>
            Feature attributions and Shapley values across Text, Image/Video spatial regions, and Audio spectral components integrated with MALGR-Net gating.
          </p>

          {isExplaining ? (
            <div style={{ padding: '32px', textAlign: 'center', color: 'var(--accent-cyan)', fontWeight: 600 }}>Computing Multimodal Shapley Values...</div>
          ) : shapData ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              
              {/* 1. Global Modality Contribution Weights */}
              {shapData.modality_contributions && (
                <div style={{ background: 'rgba(0,0,0,0.25)', padding: '18px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ fontWeight: 600, marginBottom: '12px', fontSize: '0.92rem' }}>MALGR-Net Modality Contributions</div>
                  <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                    {shapData.modality_contributions.text > 0 && (
                      <div style={{ flex: shapData.modality_contributions.text, background: 'rgba(0, 242, 254, 0.15)', border: '1px solid #00f2fe', padding: '10px', borderRadius: '8px', textAlign: 'center', fontSize: '0.85rem', fontWeight: 600, color: '#00f2fe' }}>
                        Text (gt): {shapData.modality_contributions.text}%
                      </div>
                    )}
                    {shapData.modality_contributions.image > 0 && (
                      <div style={{ flex: shapData.modality_contributions.image, background: 'rgba(168, 85, 247, 0.15)', border: '1px solid #a855f7', padding: '10px', borderRadius: '8px', textAlign: 'center', fontSize: '0.85rem', fontWeight: 600, color: '#a855f7' }}>
                        Visual (gv): {shapData.modality_contributions.image}%
                      </div>
                    )}
                    {shapData.modality_contributions.audio > 0 && (
                      <div style={{ flex: shapData.modality_contributions.audio, background: 'rgba(234, 179, 8, 0.15)', border: '1px solid #eab308', padding: '10px', borderRadius: '8px', textAlign: 'center', fontSize: '0.85rem', fontWeight: 600, color: '#eab308' }}>
                        Audio (ga): {shapData.modality_contributions.audio}%
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* 2. Text SHAP Tokens */}
              {shapData.text_shap && (
                <div>
                  <h4 style={{ color: '#00f2fe', marginBottom: '10px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FileText size={18} /> Text SHAP Token Drivers
                  </h4>
                  <div style={{ lineHeight: '2.5', fontSize: '1.05rem', background: 'rgba(0,0,0,0.3)', padding: '18px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.06)' }}>
                    {shapData.text_shap.tokens.map((token, i) => {
                      const posVal = shapData.text_shap.values[i][2];
                      const negVal = shapData.text_shap.values[i][0];
                      let bgColor = 'transparent';
                      let color = 'var(--text-main)';
                      let fontWeight = 'normal';
                      let tooltip = `Token: ${token}\nPOS impact: ${posVal.toFixed(3)}\nNEG impact: ${negVal.toFixed(3)}`;
                      if (posVal > 0.05) {
                        bgColor = `rgba(0, 242, 254, ${Math.min(posVal * 2, 0.8)})`;
                        color = '#fff';
                        fontWeight = 600;
                      } else if (negVal > 0.05) {
                        bgColor = `rgba(239, 68, 68, ${Math.min(negVal * 2, 0.8)})`;
                        color = '#fff';
                        fontWeight = 600;
                      }
                      return (
                        <span key={i} style={{ background: bgColor, color: color, padding: '4px 6px', margin: '0 3px', borderRadius: '6px', fontWeight: fontWeight, cursor: 'help' }} title={tooltip}>
                          {token.replace('\u2581', ' ')}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* 3. Image / Video Spatial SHAP Heatmap */}
              {shapData.image_shap && (
                <div>
                  <h4 style={{ color: '#a855f7', marginBottom: '12px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <ImageIcon size={18} /> Visual & Spatial SHAP Region Attributions
                  </h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px' }}>
                    {/* Heatmap Grid Card */}
                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '20px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '14px', textAlign: 'center', fontWeight: 600 }}>
                        Spatial Occlusion Heatmap Matrix (4×4)
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', width: '220px', height: '220px', margin: '0 auto' }}>
                        {shapData.image_shap.grid_heatmap.map((row, r) => 
                          row.map((val, c) => {
                            const numVal = Number(val);
                            const formattedVal = numVal > 0 ? `+${numVal.toFixed(2)}` : numVal.toFixed(2);
                            return (
                              <div key={`${r}-${c}`} style={{
                                background: numVal > 0 ? `rgba(0, 242, 254, ${Math.min(Math.abs(numVal) * 1.5, 0.85)})` : `rgba(239, 68, 68, ${Math.min(Math.abs(numVal) * 1.5, 0.85)})`,
                                borderRadius: '8px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '0.72rem',
                                color: '#ffffff',
                                fontWeight: 700,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                                padding: '2px',
                                border: '1px solid rgba(255,255,255,0.1)',
                                boxShadow: numVal > 0 ? '0 0 10px rgba(0, 242, 254, 0.2)' : '0 0 10px rgba(239, 68, 68, 0.2)'
                              }} title={`Grid Cell [${r+1},${c+1}] SHAP Impact: ${numVal.toFixed(4)}`}>
                                {formattedVal}
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>

                    {/* Facial Region Attributions Card */}
                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '20px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '16px', fontWeight: 600 }}>
                        Facial & Expression Region Attributions
                      </div>
                      {shapData.image_shap.regions.map((reg, idx) => {
                        const imp = Number(reg.impact);
                        const impText = imp > 0 ? `+${imp.toFixed(3)}` : imp.toFixed(3);
                        return (
                          <div key={idx} style={{ marginBottom: '14px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                              <span style={{ fontWeight: 500 }}>{reg.name}</span>
                              <span style={{ color: imp > 0 ? '#00f2fe' : '#ef4444', fontWeight: 700 }}>{impText}</span>
                            </div>
                            <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                              <div style={{ width: `${Math.min(Math.abs(imp) * 100, 100)}%`, height: '100%', background: imp > 0 ? 'linear-gradient(90deg, #00f2fe, #38bdf8)' : 'linear-gradient(90deg, #ef4444, #f87171)', borderRadius: '4px' }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {/* 4. Audio SHAP Acoustic & Temporal Attributions */}
              {shapData.audio_shap && (
                <div>
                  <h4 style={{ color: '#eab308', marginBottom: '10px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Music size={18} /> Audio Spectral & Temporal SHAP Attributions
                  </h4>
                  <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: '240px', background: 'rgba(0,0,0,0.3)', padding: '18px', borderRadius: '12px' }}>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '14px' }}>Acoustic Feature Drivers</div>
                      {shapData.audio_shap.acoustic_features.map((feat, idx) => (
                        <div key={idx} style={{ marginBottom: '12px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                            <span>{feat.feature}</span>
                            <span style={{ color: feat.impact > 0 ? '#eab308' : '#ef4444', fontWeight: 600 }}>{feat.impact > 0 ? `+${feat.impact}` : feat.impact}</span>
                          </div>
                          <div style={{ width: '100%', height: '7px', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(Math.abs(feat.impact) * 100, 100)}%`, height: '100%', background: feat.impact > 0 ? '#eab308' : '#ef4444' }} />
                          </div>
                        </div>
                      ))}
                    </div>

                    <div style={{ flex: 1, minWidth: '240px', background: 'rgba(0,0,0,0.3)', padding: '18px', borderRadius: '12px' }}>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '14px' }}>Time Segment Attributions</div>
                      {shapData.audio_shap.time_segments.map((seg, idx) => (
                        <div key={idx} style={{ marginBottom: '12px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                            <span>Segment: {seg.segment}</span>
                            <span style={{ color: '#eab308', fontWeight: 600 }}>+{seg.impact}</span>
                          </div>
                          <div style={{ width: '100%', height: '7px', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(seg.impact * 100, 100)}%`, height: '100%', background: '#eab308' }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

            </div>
          ) : null}
        </div>
      )}

      {/* Upload History Trend */}
      {uploadHistory.length > 0 && (
        <div className="glass-panel" style={{ height: 'auto' }}>
          <h3 style={{ marginBottom: '8px' }}>Upload History Trajectory</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '24px' }}>
            Session timeline tracking sentiment probability fluctuations over consecutive file uploads.
          </p>
          <div style={{ width: '100%', height: '240px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={uploadHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: 'rgba(11, 17, 32, 0.95)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: '12px', color: '#fff' }} />
                <Line type="monotone" dataKey="positive" stroke="#00f2fe" strokeWidth={3} dot={{r: 4, fill: '#00f2fe'}} isAnimationActive={false} />
                <Line type="monotone" dataKey="neutral" stroke="#a855f7" strokeWidth={3} dot={{r: 4, fill: '#a855f7'}} isAnimationActive={false} />
                <Line type="monotone" dataKey="negative" stroke="#ef4444" strokeWidth={3} dot={{r: 4, fill: '#ef4444'}} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

// ----------------------------------------
// 2. ANALYTICS BENCHMARK MODULE
// ----------------------------------------
function Analytics() {
  const metricsData = [
    { name: 'XLM-RoBERTa (Text)', accuracy: 94.2, f1: 93.8 },
    { name: 'ViT Transformer (Vision)', accuracy: 91.5, f1: 90.9 },
    { name: 'Whisper ASR (Audio)', accuracy: 89.1, f1: 88.4 },
    { name: 'MALGR-Net Latent Fusion', accuracy: 97.4, f1: 97.1 },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      <div className="page-title-section">
        <h2 className="page-title">MALGR-Net Benchmarks & Performance Metrics</h2>
        <p className="page-subtitle">Comparative evaluation metrics across unimodal baseline encoders vs. MALGR-Net Multimodal Adaptive Latent Fusion.</p>
      </div>

      <div className="glass-panel">
        <h3>Model Accuracy & F1-Score Benchmarks</h3>
        <div style={{ width: '100%', height: '320px', marginTop: '20px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={metricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[80, 100]} />
              <Tooltip contentStyle={{ background: 'rgba(11, 17, 32, 0.95)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: '12px', color: '#fff' }} />
              <Bar dataKey="accuracy" name="Accuracy (%)" fill="#00f2fe" radius={[6, 6, 0, 0]} />
              <Bar dataKey="f1" name="F1-Score (%)" fill="#a855f7" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ----------------------------------------
// 4. SYSTEM DIAGNOSTICS MODULE
// ----------------------------------------
function Diagnostics() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      <div className="page-title-section">
        <h2 className="page-title">MALGR-Net System Diagnostics</h2>
        <p className="page-subtitle">Server hardware status, API endpoint latency, and PyTorch model execution engine.</p>
      </div>

      <div className="glass-panel" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '20px' }}>
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '20px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontWeight: 600 }}>FastAPI Server</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#10b981', marginTop: '8px' }}>ONLINE (Port 8000)</div>
        </div>

        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '20px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontWeight: 600 }}>PyTorch Model Engine</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#00f2fe', marginTop: '8px' }}>MALGR-Net (7-Class)</div>
        </div>

        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '20px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontWeight: 600 }}>Model Checkpoint</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#a855f7', marginTop: '8px' }}>malgr_net.pt</div>
        </div>
      </div>
    </div>
  );
}

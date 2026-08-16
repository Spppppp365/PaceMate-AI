import "./App.css";
import { Analytics } from "@vercel/analytics/react";

function App() {
  return (
    <div className="app">
      <header className="navbar">
        <div className="nav-container">
          <div className="logo">
            PaceMate-AI
          </div>

          <nav>
            <a href="#research">Research</a>
            <a href="#results">Results</a>
            <a href="#demo">Demo</a>
            <a href="#methodology">Methodology</a>
          </nav>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow">
              MACHINE LEARNING RESEARCH PROJECT
            </div>

            <h1>
              Understanding symptoms <span>over time.</span>
            </h1>

            <p className="hero-description">
              PaceMate-AI is a research prototype investigating whether
              longitudinal health information can improve machine-learning
              prediction of daily symptom-related outcomes associated with
              Postural Orthostatic Tachycardia Syndrome.
            </p>

            <div className="hero-buttons">
              <a href="#demo" className="button button-primary">
                Try the Demo
              </a>

              <a href="#research" className="button button-secondary">
                Explore the Research
              </a>
            </div>

            <p className="hero-note">
              Synthetic research dataset · Not a clinical diagnostic system
            </p>
          </div>

          <div className="hero-visual">
            <div className="prediction-card main-card">
              <div className="card-header">
                <span>Daily Prediction</span>
                <span className="status-dot"></span>
              </div>

              <div className="prediction-score">
                <span>Flare Risk</span>
                <strong>68%</strong>
              </div>

              <div className="progress-track">
                <div className="progress-fill"></div>
              </div>

              <div className="prediction-details">
                <div>
                  <span>Sleep</span>
                  <strong>5.8 h</strong>
                </div>

                <div>
                  <span>HRV</span>
                  <strong>42 ms</strong>
                </div>

                <div>
                  <span>Symptoms</span>
                  <strong>Moderate</strong>
                </div>
              </div>
            </div>

            <div className="floating-card floating-one">
              <span>Longitudinal model</span>
              <strong>32 features</strong>
            </div>

            <div className="floating-card floating-two">
              <span>Participants</span>
              <strong>500</strong>
            </div>
          </div>
        </section>

        <section id="research" className="section research-section">
          <div className="section-heading">
            <div className="eyebrow">THE RESEARCH</div>

            <h2>
              Does context from the past <span>improve prediction?</span>
            </h2>

            <p>
              PaceMate-AI investigates whether information collected over
              multiple days provides predictive value beyond measurements
              taken on a single day.
            </p>
          </div>

          <div className="research-grid">
            <article className="research-card">
              <div className="research-number">01</div>
              <h3>Longitudinal information</h3>
              <p>
                Historical measurements, rolling averages, changes over time,
                and symptom trends were incorporated into the longitudinal
                model.
              </p>
            </article>

            <article className="research-card">
              <div className="research-number">02</div>
              <h3>Participant-level evaluation</h3>
              <p>
                Participants were separated between training, validation,
                and test sets so that the same participant was not shared
                across partitions.
              </p>
            </article>

            <article className="research-card">
              <div className="research-number">03</div>
              <h3>Six prediction targets</h3>
              <p>
                The models predict flare risk, dizziness risk, fatigue risk,
                fainting risk, need to hydrate, and need to rest.
              </p>
            </article>
          </div>
        </section>

        <section id="results" className="section results-section">
          <div className="section-heading centered">
            <div className="eyebrow">FINAL RESULTS</div>

            <h2>
              Longitudinal data made a <span>measurable difference.</span>
            </h2>

            <p>
              Across five repeated participant-level experiments, longitudinal
              features improved ROC-AUC, PR-AUC, and Brier score for every
              prediction target.
            </p>
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <span>Best ROC-AUC</span>
              <strong>0.9516</strong>
              <small>Fainting risk</small>
            </div>

            <div className="metric-card">
              <span>Best PR-AUC</span>
              <strong>0.7582</strong>
              <small>Need to hydrate</small>
            </div>

            <div className="metric-card">
              <span>Best F1</span>
              <strong>0.7231</strong>
              <small>Need to hydrate</small>
            </div>

            <div className="metric-card">
              <span>Participants</span>
              <strong>500</strong>
              <small>Synthetic participants</small>
            </div>
          </div>

          <div className="result-highlight">
            <div>
              <span className="highlight-label">
                Largest mean ROC-AUC improvement
              </span>
              <strong>+0.2885</strong>
              <p>
                Observed for flare risk when comparing longitudinal features
                with same-day features.
              </p>
            </div>

            <div className="highlight-divider"></div>

            <div>
              <span className="highlight-label">
                Largest mean PR-AUC improvement
              </span>
              <strong>+0.4658</strong>
              <p>
                Also observed for flare risk across the five repeated
                participant-level experiments.
              </p>
            </div>
          </div>
        </section>

        <section id="demo" className="section demo-section">
          <div className="demo-intro">
            <div className="eyebrow">INTERACTIVE DEMO</div>

            <h2>
              Explore how a prediction <span>could work.</span>
            </h2>

            <p>
              Adjust example daily measurements and explore the type of
              prediction interface being developed for PaceMate-AI.
            </p>
          </div>

          <div className="demo-placeholder">
            <div className="demo-icon">✦</div>

            <h3>Interactive prediction demo</h3>

            <p>
              This section will become an interactive research demonstration
              using the PaceMate-AI model outputs.
            </p>

            <button className="button button-primary demo-button">
              Demo coming next
            </button>
          </div>
        </section>

        <section id="methodology" className="section methodology-section">
          <div className="section-heading">
            <div className="eyebrow">METHODOLOGY</div>

            <h2>
              Designed to test <span>longitudinal prediction.</span>
            </h2>
          </div>

          <div className="methodology-grid">
            <div className="method-item">
              <strong>500</strong>
              <span>Synthetic participants</span>
            </div>

            <div className="method-item">
              <strong>180</strong>
              <span>Days per participant</span>
            </div>

            <div className="method-item">
              <strong>89,000</strong>
              <span>Modeling observations</span>
            </div>

            <div className="method-item">
              <strong>32</strong>
              <span>Longitudinal features</span>
            </div>

            <div className="method-item">
              <strong>6</strong>
              <span>Prediction targets</span>
            </div>

            <div className="method-item">
              <strong>5</strong>
              <span>Repeated experiments</span>
            </div>
          </div>
        </section>

        <section className="section limitations-section">
          <div className="limitation-box">
            <div className="eyebrow">IMPORTANT CONTEXT</div>

            <h2>What these results do and do not show.</h2>

            <p>
              PaceMate-AI uses a synthetic research dataset. The results
              demonstrate predictive behavior within a controlled machine
              learning experiment, but they do not establish clinical
              effectiveness, diagnostic accuracy, or medical safety.
            </p>

            <p>
              Real-world data, external validation, prospective evaluation,
              and clinical safety assessment would be required before drawing
              conclusions about clinical utility.
            </p>
          </div>
        </section>
      </main>

            <footer className="footer">
        <div>
          <strong>PaceMate-AI</strong>
          <span>
            A longitudinal machine-learning research prototype.
          </span>
        </div>

        <span>Research project · 2026</span>
      </footer>

      <Analytics />
    </div>
  );
}

export default App;  
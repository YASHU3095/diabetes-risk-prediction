// ==============================================================================
// DIABETESCARE AI — SCRIPT CONTROLLER (MAJOR PROJECT)
// ==============================================================================

// 1. DATE FORMATTING IN TOPBAR
const todayEl = document.getElementById("today");
if (todayEl) {
  todayEl.textContent = new Intl.DateTimeFormat("en-IN", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric"
  }).format(new Date());
}

// 2. MOBILE SIDEBAR TOGGLE & NAV SMOOTH SCROLLING
const sidebar = document.getElementById("sidebar");
const menuBtn = document.getElementById("menuBtn");
if (menuBtn && sidebar) {
  menuBtn.addEventListener("click", () => sidebar.classList.toggle("open"));
}

document.querySelectorAll(".nav-link, .brand, .button[href^='#']").forEach(link => {
  link.addEventListener("click", () => {
    if (sidebar) sidebar.classList.remove("open");
  });
});

// 3. INTERSECTION OBSERVER FOR ACTIVE NAV LINKS & BREADCRUMB
const navLinks = [...document.querySelectorAll(".nav-link")];
const sections = [...document.querySelectorAll("main section[id]")];
const crumbCurrent = document.getElementById("crumbCurrent");

const sectionTitles = {
  overview: "Overview",
  assessment: "Risk assessment",
  triage: "30% Triage studio",
  benchmarks: "Model benchmarks",
  queue: "Walk-in queue",
  lifestyle: "Lifestyle guide",
  profile: "My profile",
  help: "Help & support"
};

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const sectionId = entry.target.id;
      navLinks.forEach(link => {
        const targetHref = link.getAttribute("href");
        link.classList.toggle("active", targetHref === "#" + sectionId);
      });
      if (crumbCurrent && sectionTitles[sectionId]) {
        crumbCurrent.textContent = sectionTitles[sectionId];
      }
    }
  });
}, { rootMargin: "-25% 0px -65% 0px" });

sections.forEach(section => observer.observe(section));

// ==============================================================================
// 4. RISK ASSESSMENT FORM & MACHINE LEARNING INFERENCE
// ==============================================================================
const riskForm = document.getElementById("riskForm");
const resultPanel = document.getElementById("resultPanel");
let lastClinicalReport = null;

if (riskForm && resultPanel) {
  riskForm.addEventListener("submit", event => {
    event.preventDefault();
    if (!riskForm.reportValidity()) return;

    const data = Object.fromEntries(new FormData(riskForm).entries());

    const age = Number(data.age) || 35;
    const glucose = Number(data.glucose) || 120;
    const bp = Number(data.bp) || 72;
    const weight = Number(data.weight) || 70;
    const heightCm = Number(data.height) || 165;
    const pregnancies = Number(data.pregnancies) || 0;
    const insulin = Number(data.insulin) || 100;
    const pedigree = Number(data.pedigree) || 0.47;

    const heightM = heightCm / 100;
    const bmi = weight / (heightM * heightM);

    // Calibrated Machine Learning Log-odds calculation (matching Pima Indians trained pipeline)
    const z = -8.40 
      + (0.035 * glucose) 
      + (0.089 * bmi) 
      + (0.015 * age) 
      + (0.94 * pedigree) 
      + (0.12 * pregnancies) 
      + (0.005 * insulin);

    const probability = 1 / (1 + Math.exp(-z));
    let score = Math.round(probability * 100);
    score = Math.max(4, Math.min(96, score));

    // Clinic 30% Triage Threshold cutoff is 53.6%
    const isPriorityQueue = score >= 54;

    const label = isPriorityQueue 
      ? "PRIORITY TRIAGE FLAGGED (TOP 30% COHORT)" 
      : "ROUTINE LIFESTYLE CARE (BELOW 30% CUTOFF)";

    const title = isPriorityQueue
      ? "🚨 Route to Phlebotomy Lab for Confirmatory Blood Test"
      : "✅ Routine Lifestyle Care & Scheduled Follow-up";

    const explanation = isPriorityQueue
      ? `Patient risk score (${score}%) places them in the top 30% priority screening cohort. Under the rural clinic resource protocol, this patient is allocated an immediate laboratory blood testing kit (Fasting Blood Sugar / HbA1c venous draw). Calculated BMI: ${bmi.toFixed(1)} kg/m².`
      : `Patient risk score (${score}%) is below the clinic's 30% laboratory testing threshold. Patient can be safely reassured with dietary lifestyle counseling and routine 6-month checkup. Calculated BMI: ${bmi.toFixed(1)} kg/m².`;

    // Update Score Ring & Result Panel UI
    document.getElementById("scoreValue").textContent = score;
    document.getElementById("resultLabel").textContent = label;
    document.getElementById("resultTitle").textContent = title;
    document.getElementById("resultText").textContent = explanation;

    const ring = document.getElementById("scoreRing");
    const ringColor = isPriorityQueue ? "#c77c68" : "#557d6c";
    ring.style.background = `conic-gradient(${ringColor} ${score * 3.6}deg, #e8f0e9 0deg)`;

    // Update Overview Stats
    const statLatest = document.getElementById("statLatestAssessment");
    const statLatestFoot = document.getElementById("statLatestFoot");
    const statBmi = document.getElementById("statBmiDisplay");
    const statBmiFoot = document.getElementById("statBmiFoot");

    if (statLatest) statLatest.textContent = `${score}% Risk`;
    if (statLatestFoot) statLatestFoot.textContent = isPriorityQueue ? "Top 30% Phlebotomy queue" : "Routine lifestyle triage";
    if (statBmi) statBmi.innerHTML = `${bmi.toFixed(1)} <small>kg/m²</small>`;
    if (statBmiFoot) statBmiFoot.textContent = `Weight: ${weight}kg, Height: ${heightCm}cm`;

    // Show result panel
    resultPanel.classList.remove("hidden");
    lastClinicalReport = {
      date: new Date().toLocaleString(),
      score,
      label,
      title,
      explanation,
      bmi: bmi.toFixed(1),
      age,
      glucose,
      bp,
      pregnancies,
      insulin,
      pedigree
    };

    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

// 5. DOWNLOAD CLINICAL SUMMARY REPORT
const downloadBtn = document.getElementById("downloadReport");
if (downloadBtn) {
  downloadBtn.addEventListener("click", () => {
    if (!lastClinicalReport) return;
    const r = lastClinicalReport;
    const reportText = 
`================================================================================
AURAHEALTH AI / DIABETESCARE — CLINICAL SCREENING SUMMARY REPORT
================================================================================
Date/Time: ${r.date}
Patient Age: ${r.age} Years | Pregnancies: ${r.pregnancies}
Calculated BMI: ${r.bmi} kg/m²
Plasma Glucose: ${r.glucose} mg/dL | Blood Pressure: ${r.bp} mmHg
Serum Insulin: ${r.insulin} µU/mL | Pedigree Score: ${r.pedigree}

PREDICTED RISK SCORE: ${r.score}%
CLINICAL STATUS: ${r.label}
TRIAGE ACTION: ${r.title}

CLINICAL RATIONALE:
${r.explanation}

RURAL CLINIC 30% CAPACITY RULE:
The clinic allocates diagnostic blood tests strictly to patients whose risk 
score falls into the top 30% of cohort arrivals (Cutoff: p >= 53.6%). 

NOTICE: This is an educational screening prototype. It does not replace a doctor's 
judgment or diagnostic laboratory blood tests.
================================================================================`;

    const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `DiabetesCare-Triage-Report-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

// ==============================================================================
// 6. 30% CAPACITY TRIAGE SIMULATOR
// ==============================================================================
function updateLiveTriageSim() {
  const walkinsInput = document.getElementById("simWalkins");
  const capacityInput = document.getElementById("simCapacity");
  const modelSelect = document.getElementById("simModelSelect");

  if (!walkinsInput || !capacityInput || !modelSelect) return;

  const walkins = parseInt(walkinsInput.value, 10);
  const capacityPct = parseInt(capacityInput.value, 10);
  const model = modelSelect.value;

  document.getElementById("dispWalkins").textContent = `${walkins} Patients`;
  document.getElementById("dispCapacity").textContent = `${capacityPct}% (Fixed Quota)`;

  const prevalence = 0.35; // ~35% diabetic prevalence in walk-ins
  const totalDiab = Math.round(walkins * prevalence);
  const totalHealthy = walkins - totalDiab;
  const testsAllowed = Math.round(walkins * (capacityPct / 100));

  let baseRecall = 0.574; // KNN baseline from our test run
  let badgeName = "KNN (k=9)";
  let cutoffProb = 0.536;

  if (model === "knn") {
    baseRecall = 0.574;
    badgeName = "KNN (k=9)";
    cutoffProb = 0.536;
  } else if (model === "logreg") {
    baseRecall = 0.500;
    badgeName = "Logistic Regression";
    cutoffProb = 0.499;
  } else if (model === "dt") {
    baseRecall = 0.556;
    badgeName = "Decision Tree";
    cutoffProb = 0.557;
  } else if (model === "random") {
    baseRecall = capacityPct / 100.0;
    badgeName = "Random Guessing";
    cutoffProb = 0.500;
  }

  document.getElementById("dispModelBadge").textContent = badgeName;

  const capMultiplier = capacityPct / 30.0;
  let effectiveRecall = Math.min(0.96, baseRecall * Math.pow(capMultiplier, 0.45));
  if (model === "random") effectiveRecall = capacityPct / 100.0;

  let tp = Math.min(testsAllowed, Math.round(totalDiab * effectiveRecall));
  let fp = Math.max(0, testsAllowed - tp);
  if (fp > totalHealthy) {
    fp = totalHealthy;
    tp = testsAllowed - fp;
  }

  const fn = Math.max(0, totalDiab - tp);
  const recallPct = totalDiab > 0 ? ((tp / totalDiab) * 100).toFixed(1) : "0.0";

  document.getElementById("dispTp").textContent = tp;
  document.getElementById("dispFn").textContent = fn;
  document.getElementById("dispFp").textContent = fp;
  document.getElementById("dispRecall").textContent = `${recallPct}%`;
  document.getElementById("dispTotalDiabFoot").textContent = `out of ${totalDiab} present`;
  document.getElementById("dispTotalTestsFoot").textContent = `out of ${testsAllowed} kits`;

  const barTp = (tp / walkins) * 100;
  const barFp = (fp / walkins) * 100;
  document.getElementById("barTp").style.width = `${barTp}%`;
  document.getElementById("barFp").style.width = `${barFp}%`;

  document.getElementById("dispCutoffText").textContent = `Cutoff: p \u2265 ${cutoffProb.toFixed(3)}`;

  const narrative = document.getElementById("dispTriageNarrative");
  if (model === "random") {
    narrative.textContent = `Random triage without machine learning misses ${fn} diabetic patients daily! Only ${tp} diabetics are detected by chance.`;
  } else {
    narrative.textContent = `Sorting daily walk-in arrivals by predicted risk prioritizes high-probability patients. At ${capacityPct}% capacity, ${badgeName} captures ${recallPct}% (${tp}/${totalDiab}) of all diabetics while consuming only ${testsAllowed} test kits.`;
  }
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  updateLiveTriageSim();
});


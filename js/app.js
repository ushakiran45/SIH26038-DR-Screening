/**
 * SIH26038 - Main Application Orchestrator
 * Handles tab transitions, sample dataset selection, event listeners,
 * UI updates, and synchronization across modules.
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing SIH26038 Explainable AI DR Workstation...');

    let currentSampleKey = 'moderate_npdr';

    // 1. Initialize Tab Navigation
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            this.classList.add('active');
            const targetPane = document.getElementById(targetTab);
            if (targetPane) targetPane.classList.add('active');

            // Render specific canvas views upon tab activation
            refreshActiveTabViews(targetTab);
        });
    });

    // 2. Sample Dataset Selector Handler
    const sampleSelect = document.getElementById('sample-select');
    if (sampleSelect) {
        sampleSelect.addEventListener('change', function() {
            currentSampleKey = this.value;
            refreshAllModules();
        });
    }

    // 3. Layer Toggle Chips Handler (Segmentation View)
    const toggleChips = document.querySelectorAll('.toggle-chip');
    toggleChips.forEach(chip => {
        chip.addEventListener('click', function() {
            const layer = this.getAttribute('data-layer');
            if (layer === 'all') {
                toggleChips.forEach(c => c.classList.remove('active'));
                this.classList.add('active');
            } else {
                document.querySelector('[data-layer="all"]')?.classList.remove('active');
                this.classList.toggle('active');
            }
            renderSegmentationCanvas();
        });
    });

    // 4. Grad-CAM Controls Listener
    const gcOpacity = document.getElementById('gc-opacity');
    const gcColorMap = document.getElementById('gc-colormap');

    if (gcOpacity) {
        gcOpacity.addEventListener('input', function() {
            document.getElementById('gc-opacity-val').innerText = Math.round(this.value * 100) + '%';
            window.GradingExplainability.setOpacity(parseFloat(this.value));
            renderGradCAMCanvas();
        });
    }

    if (gcColorMap) {
        gcColorMap.addEventListener('change', function() {
            window.GradingExplainability.setColorMap(this.value);
            renderGradCAMCanvas();
        });
    }

    // 5. Fast-Track Validation Buttons
    const btnApprove = document.getElementById('btn-approve-grade');
    const btnPrintReport = document.getElementById('btn-print-report');

    if (btnApprove) {
        btnApprove.addEventListener('click', function() {
            alert('✓ Diagnosis Verified & Digitally Signed by Tele-Ophthalmologist under 30 seconds!\nRecorded in Rural Health Telemedicine Database.');
        });
    }

    if (btnPrintReport) {
        btnPrintReport.addEventListener('click', function() {
            window.GradingExplainability.printReport();
        });
    }

    // 6. Simulink Simulator Sliders Listener
    const simPop = document.getElementById('sim-pop');
    const simPhc = document.getElementById('sim-phc');
    const simBw = document.getElementById('sim-bw');
    const simDoc = document.getElementById('sim-doc');

    function triggerSimulinkUpdate() {
        if (simPop) document.getElementById('sim-pop-val').innerText = parseInt(simPop.value).toLocaleString();
        if (simPhc) document.getElementById('sim-phc-val').innerText = simPhc.value;
        if (simBw) document.getElementById('sim-bw-val').innerText = simBw.value + ' Mbps';
        if (simDoc) document.getElementById('sim-doc-val').innerText = simDoc.value;

        window.SimulinkSimulator.runSimulation({
            population: parseInt(simPop?.value || 100000),
            numPHCs: parseInt(simPhc?.value || 25),
            bandwidth: parseFloat(simBw?.value || 1.5),
            doctors: parseInt(simDoc?.value || 2)
        });
    }

    [simPop, simPhc, simBw, simDoc].forEach(slider => {
        if (slider) slider.addEventListener('input', triggerSimulinkUpdate);
    });

    // Master Refresh across all modules
    function refreshAllModules() {
        // Module 1: Quality
        window.QualityAnalyzer.updateQualityUI(currentSampleKey);
        window.FundusEngine.renderFundusImage('canvas-iqa-raw', currentSampleKey, { mode: 'raw' });
        window.FundusEngine.renderFundusImage('canvas-iqa-enhanced', currentSampleKey, { mode: 'clahe' });

        // Module 2: Segmentation
        renderSegmentationCanvas();

        // Module 3 & 4: Grading & Grad-CAM
        window.GradingExplainability.updateGradingUI(currentSampleKey);
        renderGradCAMCanvas();

        // Module 5: Simulink
        triggerSimulinkUpdate();

        // Module 6: Benchmark
        window.BenchmarkData.renderBenchmarkTable('benchmark-table-container');
    }

    function renderSegmentationCanvas() {
        const activeChips = Array.from(document.querySelectorAll('.toggle-chip.active')).map(c => c.getAttribute('data-layer'));
        const options = {
            showAllOverlay: activeChips.includes('all'),
            showOpticDisc: activeChips.includes('od'),
            showFovea: activeChips.includes('fovea'),
            showMAs: activeChips.includes('ma'),
            showExudates: activeChips.includes('exudates'),
            showHemorrhages: activeChips.includes('hemorrhages'),
            showNV: activeChips.includes('nv')
        };
        window.FundusEngine.renderFundusImage('canvas-seg-main', currentSampleKey, options);
    }

    function renderGradCAMCanvas() {
        const opacity = parseFloat(document.getElementById('gc-opacity')?.value || 0.55);
        const colormap = document.getElementById('gc-colormap')?.value || 'jet';
        window.GradingExplainability.renderGradCAM('canvas-gradcam', currentSampleKey, opacity, colormap);
    }

    function refreshActiveTabViews(tabId) {
        if (tabId === 'tab-quality') {
            window.FundusEngine.renderFundusImage('canvas-iqa-raw', currentSampleKey, { mode: 'raw' });
            window.FundusEngine.renderFundusImage('canvas-iqa-enhanced', currentSampleKey, { mode: 'clahe' });
        } else if (tabId === 'tab-segmentation') {
            renderSegmentationCanvas();
        } else if (tabId === 'tab-explainability') {
            renderGradCAMCanvas();
        } else if (tabId === 'tab-simulink') {
            triggerSimulinkUpdate();
        } else if (tabId === 'tab-benchmark') {
            window.BenchmarkData.renderBenchmarkTable('benchmark-table-container');
        }
    }

    // Initial Trigger on Page Load
    setTimeout(refreshAllModules, 100);
});

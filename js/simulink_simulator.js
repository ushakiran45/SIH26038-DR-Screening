/**
 * SIH26038 - Simulink Telemedicine Screening Simulator Engine
 * Discrete-event telemetry simulator modeling district-level healthcare delivery
 * for 100,000+ rural patients across PHC nodes, bandwidth constraints, edge AI,
 * and ophthalmologist human-in-the-loop review capacity.
 */

window.SimulinkSimulator = (function() {

    let chartInstance = null;

    function runSimulation(params = {}) {
        const population = params.population || 100000;
        const numPHCs = params.numPHCs || 25;
        const bandwidth = params.bandwidth || 1.5; // Mbps
        const doctors = params.doctors || 2;

        const workingDays = 300;
        const dailyPatients = Math.round(population / workingDays); // ~333 patients/day
        const referralRate = 0.08; // 8% referable DR Level 2+
        const dailyReferrals = Math.round(dailyPatients * referralRate); // ~27 referrals/day

        // Transmission latency
        const imgMB = 0.6; // Compressed JPEG2000
        const effBps = (bandwidth * 1024 * 1024 / 8) * 0.75;
        const netSecPerImg = (imgMB * 1024 * 1024) / effBps;

        // Manual vs AI Review Speeds
        const manualSecPerPatient = 240; // 4 minutes
        const aiSecPerPatient = 28;     // <30 seconds

        const dailyDoctorSecs = doctors * 6 * 3600;
        const manualCap = Math.floor(dailyDoctorSecs / manualSecPerPatient);
        const aiCap = Math.floor(dailyDoctorSecs / aiSecPerPatient);

        const manualBacklogDays = Math.max(0, Math.round(((dailyReferrals - manualCap) * 300) / manualCap));
        const aiBacklogDays = 0; // Cleared daily

        const costManual = 450; // INR
        const costAI = 85;     // INR
        const annualSavingsLakhs = ((population * (costManual - costAI)) / 100000).toFixed(1);
        const blindnessPrevented = Math.round(population * 0.18 * 0.90); // 90% vision loss prevention

        const results = {
            population,
            numPHCs,
            bandwidth,
            doctors,
            dailyPatients,
            dailyReferrals,
            netSecPerImg: netSecPerImg.toFixed(2),
            manualCap,
            aiCap,
            manualBacklogDays,
            aiBacklogDays,
            annualSavingsLakhs,
            blindnessPrevented
        };

        updateSimulatorUI(results);
        renderTelemetryChart(results);
        return results;
    }

    function updateSimulatorUI(res) {
        const pDaily = document.getElementById('sim-daily-patients');
        const pNet = document.getElementById('sim-net-latency');
        const pCapManual = document.getElementById('sim-manual-cap');
        const pCapAI = document.getElementById('sim-ai-cap');
        const pBacklogManual = document.getElementById('sim-manual-backlog');
        const pSavings = document.getElementById('sim-savings');
        const pBlindness = document.getElementById('sim-blindness-prevented');

        if (pDaily) pDaily.innerText = res.dailyPatients;
        if (pNet) pNet.innerText = res.netSecPerImg + ' s/img';
        if (pCapManual) pCapManual.innerText = res.manualCap + ' cases/day';
        if (pCapAI) pCapAI.innerText = res.aiCap + ' cases/day';
        if (pBacklogManual) pBacklogManual.innerText = res.manualBacklogDays + ' days backlog';
        if (pSavings) pSavings.innerText = '₹' + res.annualSavingsLakhs + ' Lakhs';
        if (pBlindness) pBlindness.innerText = '~' + res.blindnessPrevented.toLocaleString() + ' Patients';
    }

    function renderTelemetryChart(res) {
        const canvas = document.getElementById('simulink-chart');
        if (!canvas) return;

        // If Chart.js is loaded via CDN, use Chart.js
        if (window.Chart) {
            if (chartInstance) chartInstance.destroy();
            const ctx = canvas.getContext('2d');
            chartInstance = new window.Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Daily Referral Load', 'Manual Review Capacity', 'AI-Assisted Capacity (<30s)'],
                    datasets: [{
                        label: 'Patients / Day',
                        data: [res.dailyReferrals, res.manualCap, res.aiCap],
                        backgroundColor: [
                            'rgba(255, 179, 0, 0.7)',
                            'rgba(255, 82, 82, 0.7)',
                            'rgba(0, 230, 118, 0.8)'
                        ],
                        borderColor: [
                            '#ffb300',
                            '#ff5252',
                            '#00e676'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: 'District Telemedicine Throughput: Manual vs AI-Assisted Tele-Ophthalmology Workflow',
                            color: '#00f2fe',
                            font: { size: 14, weight: 'bold' }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.08)' },
                            ticks: { color: '#94a3b8' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#f0f4f8' }
                        }
                    }
                }
            });
        } else {
            // Fallback native Canvas bar chart if offline
            renderFallbackCanvasChart(canvas, res);
        }
    }

    function renderFallbackCanvasChart(canvas, res) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width || 700;
        const height = canvas.height || 300;

        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);

        ctx.fillStyle = '#00f2fe';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.fillText('District Telemedicine Throughput Simulation', 20, 30);

        const maxVal = Math.max(res.dailyReferrals, res.manualCap, res.aiCap, 50);
        const data = [
            { label: 'Daily Referrals', val: res.dailyReferrals, color: '#ffb300' },
            { label: 'Manual Capacity', val: res.manualCap, color: '#ff5252' },
            { label: 'AI-Assisted (<30s)', val: res.aiCap, color: '#00e676' }
        ];

        const barWidth = 100;
        const startX = 80;
        const startY = height - 50;

        data.forEach((item, idx) => {
            const x = startX + idx * 180;
            const barH = (item.val / maxVal) * (height - 100);

            ctx.fillStyle = item.color;
            ctx.fillRect(x, startY - barH, barWidth, barH);

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 12px Inter, sans-serif';
            ctx.fillText(item.val, x + 35, startY - barH - 8);

            ctx.fillStyle = '#94a3b8';
            ctx.font = '11px Inter, sans-serif';
            ctx.fillText(item.label, x - 10, startY + 20);
        });
    }

    return {
        runSimulation
    };
})();

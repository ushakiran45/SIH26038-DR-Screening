/**
 * SIH26139 - Benchmark Comparison Dataset & Pipeline Validation Suite
 * Provides published literature context plus this repository's measured CNN
 * backbone metrics on APTOS 2019.
 */

window.BenchmarkData = (function() {

    const BENCHMARKS = [
        {
            dataset: 'APTOS 2019 (published literature)',
            method: 'Typical ResNet-50 CNN (literature, not this repo)',
            referableSens: '~88%',
            referableSpec: '~84%',
            aucROC: '~0.91',
            maDetection: 'varies',
            explainableScore: 'Context only',
            status: 'LITERATURE'
        },
        {
            dataset: 'IDRiD (published literature)',
            method: 'Typical UNet lesion segmentation (literature, not this repo)',
            referableSens: '~84%',
            referableSpec: '~81%',
            aucROC: '~0.89',
            maDetection: 'varies',
            explainableScore: 'Context only',
            status: 'LITERATURE'
        },
        {
            dataset: 'This repository — APTOS 2019 held-out (N = 400)',
            method: 'CNN backbone EfficientNet-B3 (measured here)',
            referableSens: '88.9%',
            referableSpec: '91.6%',
            aucROC: '0.963',
            maDetection: 'not claimed',
            explainableScore: 'Decision support, not diagnosis',
            status: 'POC'
        },
        {
            dataset: 'This repository (N = 22 seed set)',
            method: 'SVM / 4-qubit VQC smoke-test (measured here)',
            referableSens: 'See Tab 7 QML table',
            referableSpec: 'See Tab 7 QML table',
            aucROC: 'not the CNN result',
            maDetection: 'not claimed',
            explainableScore: 'Proof-of-concept only',
            status: 'POC'
        }
    ];

    function renderBenchmarkTable(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        let html = `
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Dataset Benchmark</th>
                        <th>Method Architecture</th>
                        <th>Referable Sens.</th>
                        <th>Referable Spec.</th>
                        <th>ROC-AUC</th>
                        <th>MA Sub-Pixel Detection</th>
                        <th>Explainability Score</th>
                    </tr>
                </thead>
                <tbody>
        `;

        BENCHMARKS.forEach(item => {
            const isOurs = item.status === 'POC';
            html += `
                <tr class="${isOurs ? 'table-highlight' : ''}">
                    <td><strong>${item.dataset}</strong></td>
                    <td>${item.method}</td>
                    <td>${item.referableSens}</td>
                    <td>${item.referableSpec}</td>
                    <td><strong>${item.aucROC}</strong></td>
                    <td>${item.maDetection}</td>
                    <td>${item.explainableScore}</td>
                </tr>
            `;
        });

        html += `
                </tbody>
            </table>
        `;

        container.innerHTML = html;
    }

    return {
        renderBenchmarkTable
    };
})();

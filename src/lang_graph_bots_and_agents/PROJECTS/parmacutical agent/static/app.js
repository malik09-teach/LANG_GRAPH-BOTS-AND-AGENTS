document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const iterationsInput = document.getElementById('iterations');
    const btnText = runBtn.querySelector('.btn-text');
    const loader = runBtn.querySelector('.loader');
    
    const statusBadge = document.getElementById('status-badge');
    const metricsGrid = document.getElementById('metrics-grid');
    const historyContainer = document.getElementById('history-container');
    const historyList = document.getElementById('history-list');
    
    const bestScoreEl = document.getElementById('best-score');
    const bestFormulationEl = document.getElementById('best-formulation');
    const totalIterationsEl = document.getElementById('total-iterations');

    runBtn.addEventListener('click', async () => {
        const iterations = iterationsInput.value;
        
        // UI Loading state
        runBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        
        statusBadge.textContent = 'Running Agents...';
        statusBadge.className = 'status-badge running';
        
        metricsGrid.classList.remove('hidden');
        historyContainer.classList.remove('hidden');
        
        // Reset old data
        historyList.innerHTML = '';
        bestScoreEl.textContent = '--';
        bestFormulationEl.textContent = '--';
        totalIterationsEl.textContent = '--';

        try {
            const response = await fetch(`/optimize?iterations=${iterations}`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if(data.status === 'success') {
                // Update Metrics
                bestScoreEl.textContent = data.best_score.toFixed(2);
                totalIterationsEl.textContent = data.total_iterations_run;
                
                // Find best candidate
                const bestCand = data.history.find(h => h.score === data.best_score);
                if(bestCand) {
                    const [a, b, temp] = bestCand.candidate;
                    bestFormulationEl.textContent = `[Ex: ${a.toFixed(2)}, Ex: ${b.toFixed(2)}, T: ${temp.toFixed(1)}°C]`;
                }

                // Render History
                data.history.forEach((exp, idx) => {
                    const div = document.createElement('div');
                    div.className = 'history-item';
                    
                    const [a, b, temp] = exp.candidate;
                    const isHigh = exp.score > 80;
                    
                    div.innerHTML = `
                        <div class="iteration-num">#${idx + 1}</div>
                        <div class="candidate-params">
                            A: ${a.toFixed(2)} | B: ${b.toFixed(2)} | Temp: ${temp.toFixed(1)}°C
                        </div>
                        <div class="candidate-score ${isHigh ? 'high' : ''}">${exp.score.toFixed(2)}</div>
                    `;
                    historyList.appendChild(div);
                });

                statusBadge.textContent = 'Optimization Complete';
                statusBadge.className = 'status-badge completed';
            }
        } catch (error) {
            console.error(error);
            statusBadge.textContent = 'Error Occurred';
            statusBadge.className = 'status-badge error';
        } finally {
            runBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });
});

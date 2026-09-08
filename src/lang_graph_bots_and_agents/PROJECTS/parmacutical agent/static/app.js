document.addEventListener('DOMContentLoaded', () => {
    // --- Setup Variables ---
    const varsContainer = document.getElementById('variables-container');
    const addVarBtn = document.getElementById('add-var-btn');
    const startBtn = document.getElementById('start-session-btn');
    const constraintsInput = document.getElementById('constraints-input');

    // --- Sections ---
    const setupSection = document.getElementById('setup-section');
    const loopSection = document.getElementById('loop-section');
    const historySection = document.getElementById('history-section');

    // --- Loop Elements ---
    const iterCounter = document.getElementById('iteration-counter');
    const proposedParams = document.getElementById('proposed-params');
    const proposalReason = document.getElementById('proposal-reason');
    const labScoreInput = document.getElementById('lab-score');
    const submitResultBtn = document.getElementById('submit-result-btn');

    // --- History Elements ---
    const bestScoreVal = document.getElementById('best-score-val');
    const historyList = document.getElementById('history-list');

    let currentCandidate = null;
    let variablesData = [];

    // Add default row
    addVariableRow('Excipient A', 'x1', 0, 1);
    addVariableRow('Excipient B', 'x2', 0, 1);
    addVariableRow('Temperature', 'T', 20, 80);

    function addVariableRow(name='', symbol='', min=0, max=1) {
        const row = document.createElement('div');
        row.className = 'variable-row';
        row.innerHTML = `
            <input type="text" class="var-name" value="${name}" placeholder="Name">
            <input type="text" class="var-symbol" value="${symbol}" placeholder="Symbol">
            <input type="number" class="var-min" value="${min}">
            <input type="number" class="var-max" value="${max}">
            <button class="delete-btn">X</button>
        `;
        row.querySelector('.delete-btn').addEventListener('click', () => row.remove());
        varsContainer.appendChild(row);
    }

    addVarBtn.addEventListener('click', () => addVariableRow());

    // --- Start Session ---
    startBtn.addEventListener('click', async () => {
        const rows = document.querySelectorAll('.variable-row:not(.header-row)');
        variablesData = [];
        
        rows.forEach(row => {
            variablesData.push({
                name: row.querySelector('.var-name').value,
                symbol: row.querySelector('.var-symbol').value,
                min: parseFloat(row.querySelector('.var-min').value),
                max: parseFloat(row.querySelector('.var-max').value),
            });
        });

        const payload = {
            variables: variablesData,
            constraints: constraintsInput.value
        };

        const res = await fetch('/api/init', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if(res.ok) {
            setupSection.classList.add('hidden');
            loopSection.classList.remove('hidden');
            historySection.classList.remove('hidden');
            fetchNextProposal();
        }
    });

    // --- Fetch Proposal ---
    async function fetchNextProposal() {
        proposedParams.innerHTML = 'Loading...';
        labScoreInput.value = '';
        submitResultBtn.disabled = true;

        const res = await fetch('/api/propose');
        const data = await res.json();

        currentCandidate = data.candidate;
        iterCounter.textContent = `Iteration ${data.iteration}`;
        proposalReason.textContent = data.reason;

        proposedParams.innerHTML = '';
        currentCandidate.forEach((val, idx) => {
            const sym = variablesData[idx].symbol;
            const div = document.createElement('div');
            div.className = 'param-pill';
            div.textContent = `${sym} = ${val.toFixed(3)}`;
            proposedParams.appendChild(div);
        });

        submitResultBtn.disabled = false;
        labScoreInput.focus();
    }

    // --- Submit Result ---
    submitResultBtn.addEventListener('click', async () => {
        const score = parseFloat(labScoreInput.value);
        if(isNaN(score)) { alert("Please enter a valid number"); return; }

        submitResultBtn.disabled = true;

        const payload = {
            candidate: currentCandidate,
            score: score
        };

        await fetch('/api/report', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        updateHistory(currentCandidate, score);
        fetchNextProposal();
    });

    let bestScore = -Infinity;
    let iterationCount = 0;

    function updateHistory(candidate, score) {
        iterationCount++;
        if(score > bestScore) {
            bestScore = score;
            bestScoreVal.textContent = bestScore.toFixed(2);
        }

        const div = document.createElement('div');
        div.className = 'history-item';
        
        let paramStr = candidate.map((v, i) => `${variablesData[i].symbol}: ${v.toFixed(3)}`).join(' | ');
        
        div.innerHTML = `
            <span>[#${iterationCount}] ${paramStr}</span>
            <span class="history-score">${score.toFixed(2)}</span>
        `;
        historyList.prepend(div);
    }
});

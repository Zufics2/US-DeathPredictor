document.addEventListener("DOMContentLoaded", async () => {
    try {
        const metaRes = await axios.get('/meta');
        const stateSelect = document.getElementById('state');
        const causeSelect = document.getElementById('cause');

        stateSelect.innerHTML = metaRes.data.states.map(s => `<option value="${s}">${s}</option>`).join('');
        causeSelect.innerHTML = metaRes.data.causes.map(c => `<option value="${c}">${c}</option>`).join('');

        const metricsRes = await axios.get('/metrics');
        document.getElementById('metric-r2').innerText = metricsRes.data.r2.toFixed(4);
        document.getElementById('metric-mae').innerText = metricsRes.data.mae.toFixed(2);
        document.getElementById('metric-rmse').innerText = metricsRes.data.rmse.toFixed(2);

    } catch (err) {
        console.error("Ошибка при инициализации фронтенда:", err);
    }
});

document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const payload = {
        year: parseInt(document.getElementById('year').value),
        state: document.getElementById('state').value,
        cause_name: document.getElementById('cause').value
    };

    try {
        const res = await axios.post('/predict', payload);
        
        document.getElementById('predictedValue').innerText = res.data.predicted_age_adjusted_death_rate;
        document.getElementById('resultBlock').classList.remove('hidden');
    } catch (err) {
        alert("Ошибка при запросе предсказания. Проверь консоль бэкенда.");
        console.error(err);
    }
});
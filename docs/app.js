const model = window.HOUSE_MODEL;
const fields = ['bedroom', 'bathroom', 'age'];
const elements = {};
['area', 'locality', ...fields, 'price', 'range', 'warning', 'accuracy'].forEach((id) => {
    elements[id] = document.getElementById(id);
});

// The same arithmetic as the trained model; tests/test_model.py checks they agree.
function predictPrice(input) {
    let value = model.intercept;
    value += model.area.coef * (Math.log(input.area) - model.area.mean) / model.area.scale;
    fields.forEach((name) => {
        const part = model.numeric[name];
        const raw = Number.isFinite(input[name]) ? input[name] : part.median;
        value += part.coef * (raw - part.mean) / part.scale;
    });
    value += model.localities[input.locality] || 0;
    return Math.exp(value);
}

function formatPrice(price) {
    if (price >= 10000000) {
        return `₹${(price / 10000000).toFixed(2)} crore`;
    }
    return `₹${(price / 100000).toFixed(1)} lakh`;
}

function readNumber(id) {
    return elements[id].value === '' ? NaN : Number(elements[id].value);
}

function outOfRange(input) {
    const limits = [['area', model.area], ...fields.map((name) => [name, model.numeric[name]])];
    return limits.filter(([name, limit]) => Number.isFinite(input[name]) && (input[name] < limit.min || input[name] > limit.max))
        .map(([name]) => name);
}

function update() {
    const input = { area: readNumber('area'), locality: elements.locality.value };
    fields.forEach((name) => {
        input[name] = readNumber(name);
    });

    if (!Number.isFinite(input.area) || input.area <= 0) {
        elements.price.textContent = '—';
        elements.range.textContent = 'Enter an area above zero.';
        elements.warning.hidden = true;
        return;
    }

    const price = predictPrice(input);
    const margin = model.metrics.mape;
    elements.price.textContent = formatPrice(price);
    elements.range.textContent = `Likely between ${formatPrice(price * (1 - margin))} and ${formatPrice(price * (1 + margin))}`;

    const outside = outOfRange(input);
    elements.warning.hidden = outside.length === 0;
    elements.warning.textContent = outside.length
        ? `The ${outside.join(', ')} you entered is outside what the model learned from, so treat this as a rough guess.`
        : '';
}

Object.keys(model.localities).forEach((name) => {
    elements.locality.add(new Option(name, name));
});
elements.locality.value = 'Riverview';

elements.accuracy.textContent = `On houses it hadn't seen, the model's prices were off by about ${Math.round(model.metrics.mape * 100)}% on average.`;

document.getElementById('form').addEventListener('input', update);
document.getElementById('form').addEventListener('submit', (event) => event.preventDefault());
update();

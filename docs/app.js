const model = window.HOUSE_MODEL;
const apiUrl = window.HOUSE_API;
const REQUEST_DELAY_MS = 150;
const fields = ['bedroom', 'bathroom', 'age'];
const elements = {};
['area', 'locality', ...fields, 'price', 'range', 'warning', 'accuracy', 'source'].forEach((id) => {
    elements[id] = document.getElementById(id);
});

let latestRequest = 0;
let requestTimer = null;

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

function readInput() {
    const input = { area: readNumber('area'), locality: elements.locality.value };
    fields.forEach((name) => {
        input[name] = readNumber(name);
    });
    return input;
}

function outOfRange(input) {
    const limits = [['area', model.area], ...fields.map((name) => [name, model.numeric[name]])];
    return limits.filter(([name, limit]) => Number.isFinite(input[name]) && (input[name] < limit.min || input[name] > limit.max))
        .map(([name]) => name);
}

function showResult(price, low, high, warning) {
    elements.price.textContent = formatPrice(price);
    elements.range.textContent = `About 8 in 10 similar houses: ${formatPrice(low)} to ${formatPrice(high)}`;
    elements.warning.hidden = !warning;
    elements.warning.textContent = warning;
}

function showMessage(message) {
    elements.price.textContent = '—';
    elements.range.textContent = message;
    elements.warning.hidden = true;
}

function updateInBrowser(input) {
    const price = predictPrice(input);
    const outside = outOfRange(input);
    const warning = outside.length
        ? `The ${outside.join(', ')} you entered is outside what the model learned from, so treat this as a rough guess.`
        : '';
    showResult(price, price * model.priceRange.low, price * model.priceRange.high, warning);
}

async function updateFromServer(input) {
    const request = ++latestRequest;
    const body = { area: input.area, locality: input.locality };
    fields.forEach((name) => {
        body[name] = Number.isFinite(input[name]) ? input[name] : null;
    });

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const result = await response.json();
        if (request !== latestRequest) {
            return; // A newer change has already been sent, so this answer is out of date.
        }
        if (!response.ok) {
            showMessage(Object.values(result.errors || {})[0] || 'The server could not read those details.');
            return;
        }
        showResult(result.predicted_price, result.low, result.high, result.warnings.join(' '));
    } catch (error) {
        if (request === latestRequest) {
            showMessage("The server isn't responding. Check that it's running and try again.");
        }
    }
}

function update() {
    const input = readInput();
    clearTimeout(requestTimer);
    if (!Number.isFinite(input.area) || input.area <= 0) {
        latestRequest++; // Ignore any answer still on its way.
        showMessage('Enter an area above zero.');
    } else if (apiUrl) {
        requestTimer = setTimeout(() => updateFromServer(input), REQUEST_DELAY_MS);
    } else {
        updateInBrowser(input);
    }
}

Object.keys(model.localities).forEach((name) => {
    elements.locality.add(new Option(name, name));
});
elements.locality.value = 'Riverview';

elements.accuracy.textContent = `On houses it hadn't seen, the model's prices were off by about ${Math.round(model.metrics.mape * 100)}% on average.`;
elements.source.textContent = apiUrl ? 'Prices come from the FastAPI server.' : 'Prices are calculated in your browser.';

document.getElementById('form').addEventListener('input', update);
document.getElementById('form').addEventListener('submit', (event) => event.preventDefault());
update();

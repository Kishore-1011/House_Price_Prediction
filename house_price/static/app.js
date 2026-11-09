const REQUEST_DELAY_MS = 150;
const DEFAULT_LOCALITY = 'Riverview';
const SERVER_DOWN = "The server isn't responding. Check that it's running and try again.";
const fields = ['bedroom', 'bathroom', 'age'];
const elements = {};
['area', 'locality', ...fields, 'price', 'range', 'warning', 'accuracy'].forEach((id) => {
    elements[id] = document.getElementById(id);
});

let latestRequest = 0;
let requestTimer = null;

function formatPrice(price) {
    if (price >= 10000000) {
        return `₹${(price / 10000000).toFixed(2)} crore`;
    }
    return `₹${(price / 100000).toFixed(1)} lakh`;
}

function readNumber(id) {
    return elements[id].value === '' ? null : Number(elements[id].value);
}

function readInput() {
    const input = { area: readNumber('area'), locality: elements.locality.value };
    fields.forEach((name) => {
        input[name] = readNumber(name);
    });
    return input;
}

function showResult(result) {
    elements.price.textContent = formatPrice(result.predicted_price);
    elements.range.textContent = `${result.range_share}: ${formatPrice(result.low)} to ${formatPrice(result.high)}`;
    elements.warning.hidden = result.warnings.length === 0;
    elements.warning.textContent = result.warnings.join(' ');
}

function showMessage(message) {
    elements.price.textContent = '—';
    elements.range.textContent = message;
    elements.warning.hidden = true;
}

async function requestPrice(input) {
    const request = ++latestRequest;
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(input),
        });
        const result = await response.json();
        if (request !== latestRequest) {
            return; // A newer change has already been sent, so this answer is out of date.
        }
        if (response.ok) {
            showResult(result);
        } else {
            showMessage(Object.values(result.errors || {})[0] || 'The server could not read those details.');
        }
    } catch (error) {
        if (request === latestRequest) {
            showMessage(SERVER_DOWN);
        }
    }
}

function update() {
    const input = readInput();
    clearTimeout(requestTimer);
    if (input.area === null || input.area <= 0) {
        latestRequest++; // Ignore any answer still on its way.
        showMessage('Enter an area above zero.');
    } else {
        requestTimer = setTimeout(() => requestPrice(input), REQUEST_DELAY_MS);
    }
}

async function start() {
    try {
        const response = await fetch('/options');
        if (!response.ok) {
            throw new Error('Options unavailable');
        }
        const options = await response.json();
        options.localities.forEach((name) => {
            elements.locality.add(new Option(name, name));
        });
        elements.locality.value = DEFAULT_LOCALITY;
        elements.accuracy.textContent = `On houses it hadn't seen, the model's prices were off by about ${options.typical_error_percent}% on average.`;
        update();
    } catch (error) {
        showMessage(SERVER_DOWN);
    }
}

document.getElementById('form').addEventListener('input', update);
document.getElementById('form').addEventListener('submit', (event) => event.preventDefault());
start();

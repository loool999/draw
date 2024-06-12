const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
const guessList = document.getElementById('guess-list');
const labelInput = document.getElementById('label-input');
const trainButton = document.getElementById('train-button');

let drawing = false;

ctx.lineWidth = 2;
ctx.strokeStyle = "#FFFFFF";

canvas.addEventListener('mousedown', (e) => {
    if (e.button === 0) {
        drawing = true;
        ctx.beginPath();
        ctx.moveTo(e.offsetX, e.offsetY);
    }
});

canvas.addEventListener('mousemove', (e) => {
    if (drawing && e.buttons === 1) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    } else if (e.buttons === 2) {
        ctx.clearRect(e.offsetX, e.offsetY, 10, 10);
    }
});

canvas.addEventListener('mouseup', () => {
    if (drawing) {
        drawing = false;
        ctx.closePath();
        sendDrawing();
    }
});

canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'c' || e.key === 'C') {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
});

trainButton.addEventListener('click', () => {
    const label = labelInput.value.trim();
    if (label) {
        sendDrawing(label);
    } else {
        alert("Please enter a label.");
    }
});

function sendDrawing(label = null) {
    const dataURL = canvas.toDataURL();
    const endpoint = label ? '/train' : '/predict';
    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: dataURL, label: label }),
    })
    .then(response => response.json())
    .then(data => {
        if (label) {
            fetch('/check_word', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ word: label }),
            })
            .then(response => response.json())
            .then(result => {
                if (result.exists) {
                    alert("No");
                } else {
                    alert("Training complete.");
                }
            });
            } else {
                guessList.innerHTML = '';
                data.guesses.forEach(guess => {
                    const li = document.createElement('li');
                    li.textContent = `${guess.label}`;
                    const span = document.createElement('span');
                    span.textContent = `${guess.confidence.toFixed(4)}%`;
                    li.appendChild(span);
                    guessList.appendChild(li);
                });
            }
        });
}
labelInput.addEventListener('keydown', (e) => {
    if (e.key === 'c') {
        e.stopPropagation();
    }
});
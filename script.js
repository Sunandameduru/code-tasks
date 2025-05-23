let startTime = 0;
let updatedTime = 0;
let difference = 0;
let timerInterval;
let isRunning = false;

function start() {
  if (!isRunning) {
    startTime = new Date().getTime() - difference;
    timerInterval = setInterval(updateDisplay, 100);
    isRunning = true;
  }
}

function pause() {
  clearInterval(timerInterval);
  isRunning = false;
}

function reset() {
  clearInterval(timerInterval);
  document.getElementById('display').textContent = "00:00:00";
  document.getElementById('laps').innerHTML = '';
  difference = 0;
  isRunning = false;
}

function lap() {
  if (isRunning) {
    const lapTime = document.getElementById('display').textContent;
    const lapItem = document.createElement('li');
    lapItem.textContent = lapTime;
    document.getElementById('laps').appendChild(lapItem);
  }
}

function updateDisplay() {
  updatedTime = new Date().getTime();
  difference = updatedTime - startTime;

  let hrs = Math.floor(difference / 3600000);
  let mins = Math.floor((difference % 3600000) / 60000);
  let secs = Math.floor((difference % 60000) / 1000);

  document.getElementById('display').textContent =`${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
}

function pad(unit) {
  return unit < 10 ? '0' + unit : unit;
}
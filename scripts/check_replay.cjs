// Dependency-free behavioral smoke test for our generated offline replay.
// This checks JavaScript controls; it is not a visual browser test.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const page = fs.readFileSync(process.argv[2] || 'runs/pilot-replay.html', 'utf8');
const source = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = {};
for (const id of ['board', 'move', 'seek', 'play', 'prev', 'next', 'speed']) {
  elements[id] = { value: id === 'speed' ? '1000' : '0' };
}
const state = { hidden: false, tick: null };
const sandbox = {
  document: {
    getElementById: id => elements[id],
    body: { classList: { toggle: () => { state.hidden = !state.hidden; } } }
  },
  setInterval: fn => { state.tick = fn; return 1; },
  clearInterval: () => { state.tick = null; }
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
assert.equal(elements.move.textContent, 'Starting position');
assert.ok(elements.board.innerHTML.includes('<svg'));
elements.next.onclick();
assert.notEqual(elements.move.textContent, 'Starting position');
elements.prev.onclick();
assert.equal(elements.move.textContent, 'Starting position');
elements.play.onclick();
assert.equal(elements.play.textContent, 'Pause');
state.tick();
assert.notEqual(elements.move.textContent, 'Starting position');
elements.play.onclick();
assert.equal(elements.play.textContent, 'Play');
sandbox.document.onkeydown({key:'h',target:{tagName:'BODY'}});
assert.equal(state.hidden, true);
console.log('Replay controls passed; SVG board frames are embedded.');

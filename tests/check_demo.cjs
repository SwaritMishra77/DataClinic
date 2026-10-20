// Minimal DOM check for the committed standalone page. No browser download needed.
const fs = require('fs');
const vm = require('vm');

const source = fs.readFileSync(process.argv[2] || 'docs/index.html', 'utf8');
const code = source.split('<script>')[1].split('</script>')[0];
if (!code || code.includes('__DATACLINIC_DATA__')) throw Error('Missing embedded results');

class Element {
  constructor(id = '') {
    this.id = id; this.children = []; this.style = {}; this.textContent = '';
    this.value = ''; this.events = {}; this.disabled = false;
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren() { this.children = []; }
  setAttribute() {}
  addEventListener(type, handler) { this.events[type] = handler; }
}

const ids = ['dataset','mode','method','budget','comparison','precision','found',
  'gain','base','queue','queue-note'];
const elements = Object.fromEntries(ids.map(id => [id, new Element(id)]));
Object.assign(elements.dataset, {value:'digits'});
Object.assign(elements.mode, {value:'boundary'});
Object.assign(elements.method, {value:'rank_fusion'});
Object.assign(elements.budget, {value:'100'});
const document = {
  getElementById: id => elements[id],
  createElement: () => new Element(),
};

vm.runInNewContext(code, {document, console});
if (elements.comparison.children.length !== 6 || elements.queue.children.length !== 30)
  throw Error('Initial comparison or visual queue is incomplete');
const first = elements.queue.children[0];
first.children[1].children[2].events.click();
if (!first.children[1].children[3].textContent) throw Error('Review action failed');
elements.dataset.value = 'breast_cancer';
elements.dataset.events.change();
if (elements.queue.children.length !== 0 || !elements.precision.textContent.includes('%'))
  throw Error('Tabular dataset switch failed');
console.log('Demo interaction check passed');

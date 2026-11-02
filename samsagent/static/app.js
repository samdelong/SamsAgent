const $ = id => document.getElementById(id);
let current = null, stream = null, selectedAgent = null, rootAgent = null, refreshTimer = null;
let lastMessage = 0, activity = [];
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(typeof error.detail === 'string' ? error.detail : `Request failed (${response.status})`); }
  return response.json();
}
function showError(error) { $('error').hidden = false; $('error').textContent = error.message; }
function safe(fn) { return (...args) => { if(args[0]?.type === 'submit') args[0].preventDefault(); return Promise.resolve().then(() => fn(...args)).catch(showError); }; }
function element(tag, text, className) { const node = document.createElement(tag); node.textContent = text; if(className) node.className = className; return node; }
async function listSessions() {
  const sessions = await api('/api/sessions'); $('sessions').replaceChildren();
  for (const session of sessions) { const button = element('button', session.goal, session.id === current ? 'active' : ''); button.title = `${session.status}: ${session.goal}`; button.onclick = safe(() => openSession(session.id)); $('sessions').append(button); }
}
async function loadMessages() {
  if (!selectedAgent) return;
  const agentID = selectedAgent;
  const data = await api(`/api/agents/${agentID}?after=${lastMessage}`);
  if (selectedAgent !== agentID) return;
  for (const message of data.messages) {
    lastMessage = message.id;
    if (!['user','assistant'].includes(message.role) || !message.content) continue;
    const node = element('div','',`message ${message.role}`);
    node.append(element('small', message.role === 'user' ? 'Instruction / update' : data.role));
    node.append(document.createTextNode(message.content)); $('messages').append(node);
  }
}
async function refresh() {
  if (!current) return;
  const sessionID = current, data = await api(`/api/sessions/${sessionID}`);
  if (current !== sessionID) return;
  $('title').textContent = data.goal; $('status').textContent = data.status;
  $('usage').textContent = `${data.mode} · ${data.tokens.toLocaleString()} tokens · ${Math.floor(data.active_seconds/60)} active min`;
  $('notes').textContent = data.state;
  rootAgent = data.agents.find(a => a.role === 'coordinator')?.id;
  if (!selectedAgent) selectedAgent = rootAgent;
  $('agents').replaceChildren();
  for (const agent of data.agents) {
    const node = element('div','','agent');
    const button = element('button', `${'↳ '.repeat(agent.depth)}${agent.role} · ${agent.id.slice(0,6)}`);
    button.onclick = safe(async () => { selectedAgent=agent.id; lastMessage=0; $('messages').replaceChildren(); await loadMessages(); });
    node.append(button, element('small', `${agent.status} — ${agent.task.slice(0,110)}`)); $('agents').append(node);
  }
  $('workflows').replaceChildren();
  for(const flow of data.workflows) { const node=element('div',flow.id.slice(0,8),'flow'); node.append(element('small',`${flow.status} ${flow.result.slice(0,150)}`)); $('workflows').append(node); }
  if(!data.workflows.length) $('workflows').append(element('small','No workflows yet.'));
  $('artifacts').replaceChildren();
  for(const path of data.artifacts) { const a=element('a',path); a.href=`/api/sessions/${current}/artifacts/${path.split('/').map(encodeURIComponent).join('/')}`; a.download=''; $('artifacts').append(a); }
  await loadMessages();
}
async function openSession(id) {
  if(stream) stream.close(); current=id; selectedAgent=null; lastMessage=0; activity=[];
  $('messages').replaceChildren(); $('activity').textContent=''; $('setup').hidden=true; $('workspace').hidden=false; $('error').hidden=true;
  await refresh(); await listSessions();
  stream = new EventSource(`/api/sessions/${id}/events`);
  stream.onopen = () => $('connection').textContent='● Connected';
  stream.onerror = () => $('connection').textContent='Reconnecting…';
  stream.onmessage = event => {
    if(current !== id) return;
    const item=JSON.parse(event.data); activity.push(`${new Date(item.created*1000).toLocaleTimeString()} ${item.kind} ${JSON.stringify(item.data).slice(0,700)}`);
    activity=activity.slice(-200); $('activity').textContent=activity.join('\n');
    clearTimeout(refreshTimer); refreshTimer=setTimeout(safe(refresh),200);
  };
}
$('new').onclick=()=>{if(stream) stream.close(); current=null; $('setup').hidden=false; $('workspace').hidden=true; $('title').textContent='What are we investigating?'; $('status').textContent='Ready'; $('error').hidden=true;};
$('create').onsubmit=safe(async event=>{event.preventDefault();const result=await api('/api/sessions',{goal:$('goal').value,criteria:$('criteria').value,mode:$('goalMode').checked?'goal':'chat',token_budget:$('tokenBudget').value?Number($('tokenBudget').value):null,time_budget:$('timeBudget').value?Number($('timeBudget').value)*60:null});await openSession(result.id);});
$('followup').onsubmit=safe(async event=>{event.preventDefault();await api(`/api/sessions/${current}/messages`,{text:$('message').value,agent_id:rootAgent});$('message').value='';await refresh();});
for(const button of document.querySelectorAll('[data-control]')) button.onclick=safe(async()=>{await api(`/api/sessions/${current}/control`,{action:button.dataset.control});await refresh();});
safe(async()=>{const health=await api('/api/health');$('connection').textContent=`● ${health.model}`;await listSessions();})();

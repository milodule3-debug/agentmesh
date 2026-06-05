import { useState, useRef, useCallback, useEffect } from "react";

const N = {
  n0:"#2E3440",n1:"#3B4252",n2:"#434C5E",n3:"#4C566A",
  n4:"#D8DEE9",n5:"#E5E9F0",n6:"#ECEFF4",
  f1:"#8FBCBB",f2:"#88C0D0",f3:"#81A1C1",f4:"#5E81AC",
  a1:"#BF616A",a2:"#D08770",a3:"#EBCB8B",a4:"#A3BE8C",a5:"#B48EAD",
};

const ND = {
  trigger: {label:"Manual Trigger",color:N.a4,icon:"▶",cat:"flow",ins:0,outs:1},
  schedule:{label:"Schedule",      color:N.a4,icon:"⏱",cat:"flow",ins:0,outs:1},
  webhook: {label:"Webhook",       color:N.f2,icon:"⇄",cat:"flow",ins:0,outs:1},
  ai:      {label:"AI Agent",      color:N.f2,icon:"◈",cat:"ai",  ins:1,outs:1},
  http:    {label:"HTTP Request",  color:N.a3,icon:"⟳",cat:"action",ins:1,outs:1},
  scraper: {label:"Web Scraper",   color:N.a3,icon:"⌗",cat:"action",ins:1,outs:1},
  subflow: {label:"Sub-workflow",  color:N.f3,icon:"⊛",cat:"action",ins:1,outs:1},
  code:    {label:"Code (JS)",     color:N.a5,icon:"{}",cat:"action",ins:1,outs:1},
  text:    {label:"Text Template", color:N.a2,icon:"T", cat:"transform",ins:1,outs:1},
  file:    {label:"File Input",    color:N.a1,icon:"⊞",cat:"input",ins:0,outs:1},
  split:   {label:"Split Text",    color:N.f1,icon:"⑂",cat:"transform",ins:1,outs:2},
  merge:   {label:"Merge",         color:N.f3,icon:"⑁",cat:"transform",ins:2,outs:1},
  filter:  {label:"Filter / If",   color:N.f2,icon:"⋈",cat:"transform",ins:1,outs:2},
  loop:    {label:"Loop",          color:N.a3,icon:"↻",cat:"transform",ins:1,outs:1},
  setvar:  {label:"Set Variable",  color:N.f3,icon:"$=",cat:"vars",ins:1,outs:1},
  getvar:  {label:"Get Variable",  color:N.f3,icon:"$?",cat:"vars",ins:0,outs:1},
  notify:  {label:"Notify",        color:N.a5,icon:"🔔",cat:"output",ins:1,outs:0},
  note:    {label:"Note",          color:N.a3,icon:"✎",cat:"util",ins:0,outs:0},
  output:  {label:"Output",        color:N.a1,icon:"⊕",cat:"output",ins:1,outs:0},
};

const CATS = {
  flow:{label:"Flow",color:N.a4}, input:{label:"Input",color:N.a1},
  ai:{label:"AI",color:N.f2}, action:{label:"Actions",color:N.a3},
  transform:{label:"Transform",color:N.f1}, vars:{label:"Variables",color:N.f3},
  util:{label:"Utility",color:N.a3}, output:{label:"Output",color:N.a2},
};

const PROVS = {
  // ── Frontier
  openai:     {name:"OpenAI",        ep:"https://api.openai.com/v1/chat/completions",                                      models:["gpt-4o","gpt-4o-mini","gpt-4-turbo","o1-mini","o3-mini"],                                      def:"gpt-4o",       cat:"frontier", color:"#74AA9C"},
  anthropic:  {name:"Anthropic",     ep:"https://api.anthropic.com/v1/messages",                                           models:["claude-sonnet-4-5","claude-opus-4-5","claude-haiku-4-5-20251001"],                             def:"claude-sonnet-4-5",cat:"frontier",color:N.a2,native:true},
  gemini:     {name:"Google Gemini", ep:"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",        models:["gemini-2.0-flash","gemini-1.5-pro-latest","gemini-1.5-flash-latest"],                          def:"gemini-2.0-flash",cat:"frontier",color:"#4285F4"},
  xai:        {name:"xAI / Grok",    ep:"https://api.x.ai/v1/chat/completions",                                            models:["grok-3","grok-3-mini","grok-beta"],                                                            def:"grok-3-mini",  cat:"frontier", color:"#ffffff"},
  // ── Fast & Open
  deepseek:   {name:"DeepSeek",      ep:"https://api.deepseek.com/v1/chat/completions",                                    models:["deepseek-chat","deepseek-reasoner"],                                                           def:"deepseek-chat",cat:"fast",    color:N.a4},
  groq:       {name:"Groq",          ep:"https://api.groq.com/openai/v1/chat/completions",                                 models:["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"],          def:"llama-3.3-70b-versatile",cat:"fast",color:N.a5},
  together:   {name:"Together AI",   ep:"https://api.together.xyz/v1/chat/completions",                                    models:["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo","Qwen/Qwen2.5-72B-Instruct-Turbo","mistralai/Mixtral-8x7B-Instruct-v0.1"], def:"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",cat:"fast",color:N.f3},
  fireworks:  {name:"Fireworks AI",  ep:"https://api.fireworks.ai/inference/v1/chat/completions",                         models:["accounts/fireworks/models/llama-v3p1-70b-instruct","accounts/fireworks/models/firefunction-v2"],def:"accounts/fireworks/models/llama-v3p1-70b-instruct",cat:"fast",color:"#FF6B35"},
  // ── Specialized
  mistral:    {name:"Mistral AI",    ep:"https://api.mistral.ai/v1/chat/completions",                                      models:["mistral-large-latest","mistral-small-latest","codestral-latest","open-mixtral-8x22b"],         def:"mistral-large-latest",cat:"specialized",color:"#FF7000"},
  cohere:     {name:"Cohere",        ep:"https://api.cohere.com/compatibility/v1/chat/completions",                        models:["command-r-plus","command-r","command-a-03-2025"],                                              def:"command-r-plus",cat:"specialized",color:"#39594D"},
  perplexity: {name:"Perplexity",    ep:"https://api.perplexity.ai/chat/completions",                                      models:["llama-3.1-sonar-large-128k-online","llama-3.1-sonar-small-128k-online","sonar-pro"],           def:"llama-3.1-sonar-large-128k-online",cat:"specialized",color:"#20808D"},
  cerebras:   {name:"Cerebras",      ep:"https://api.cerebras.ai/v1/chat/completions",                                     models:["llama3.1-8b","llama3.1-70b","llama-3.3-70b"],                                                 def:"llama3.1-70b", cat:"specialized",color:"#6C5CE7"},
  // ── Gateways
  openrouter: {name:"OpenRouter",    ep:"https://openrouter.ai/api/v1/chat/completions",                                   models:["openai/gpt-4o","anthropic/claude-3.5-sonnet","google/gemini-2.0-flash-001","meta-llama/llama-3.1-405b-instruct","deepseek/deepseek-r1"], def:"openai/gpt-4o",cat:"gateway",color:N.f2},
  huggingface:{name:"HuggingFace",   ep:"https://api-inference.huggingface.co/v1/chat/completions",                        models:["meta-llama/Meta-Llama-3.1-8B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","Qwen/Qwen2.5-72B-Instruct"], def:"meta-llama/Meta-Llama-3.1-8B-Instruct",cat:"gateway",color:"#FFD21E"},
  // ── Local
  ollama:     {name:"Ollama (Local)",ep:"http://localhost:11434/v1/chat/completions",                                      models:["llama3.2","codellama","qwen2.5-coder","mistral","phi4","deepseek-coder-v2"],                    def:"llama3.2",     cat:"local",    color:N.a4, noKey:true},
};

const PROV_CATS = {
  frontier:   {label:"Frontier Models", color:"#74AA9C"},
  fast:       {label:"Fast & Open",     color:N.a4},
  specialized:{label:"Specialized",     color:N.f2},
  gateway:    {label:"Gateways",        color:N.f3},
  local:      {label:"Local",           color:N.a3},
};

const NW=234, NH=54, PR=8;
let _ctr=0;
const uid = () => `n_${Date.now()}_${_ctr++}`;

const subVars = (s, v) => (s||"").replace(/\{\{(\w+)\}\}/g, (_,k) => v&&v[k]!==undefined ? v[k] : `{{${k}}}`);

function topoSort(nodes, edges) {
  const inE = {};
  edges.forEach(e => { (inE[e.tn] = inE[e.tn] || []).push(e.fn); });
  const visited = new Set(), order = [];
  const visit = id => { if (visited.has(id)) return; visited.add(id); (inE[id]||[]).forEach(visit); order.push(id); };
  nodes.forEach(n => visit(n.id));
  return order;
}

const outP = (n, i=0) => {
  const k = ND[n.type]?.outs || 1;
  return k===1 ? {x:n.x+NW, y:n.y+NH/2} : {x:n.x+NW, y:n.y+NH/2+(i-(k-1)/2)*22};
};
const inP = (n, i=0) => {
  const k = ND[n.type]?.ins || 1;
  return k<=1 ? {x:n.x, y:n.y+NH/2} : {x:n.x, y:n.y+NH/2+(i-(k-1)/2)*22};
};
const bez = (a, b) => {
  const dx = Math.max(Math.abs(b.x-a.x)*.55, 60);
  return `M${a.x},${a.y} C${a.x+dx},${a.y} ${b.x-dx},${b.y} ${b.x},${b.y}`;
};

function defCfg(t) {
  if (t==="ai")       return {provider:"deepseek",model:"deepseek-chat",systemPrompt:"You are a helpful assistant.",temperature:0.7,maxTokens:1024,inputVar:"{{input}}"};
  if (t==="http")     return {url:"https://",method:"GET",body:""};
  if (t==="code")     return {code:"// input available\nreturn input.toUpperCase();"};
  if (t==="text")     return {template:"Summarize:\n\n{{input}}\n\nProvide 3 key takeaways."};
  if (t==="file")     return {content:"",filename:""};
  if (t==="split")    return {delimiter:"\\n\\n",maxParts:10};
  if (t==="filter")   return {condition:"input.length > 0"};
  if (t==="merge")    return {separator:"\\n---\\n"};
  if (t==="trigger")  return {triggerData:"Research the latest trends in AI agents."};
  if (t==="schedule") return {cron:"0 9 * * 1-5"};
  if (t==="webhook")  return {simPayload:'{"event":"trigger","data":"hello"}',extractField:"",port:8000};
  if (t==="subflow")  return {workflowJson:null,workflowName:"(none)",passVars:false};
  if (t==="loop")     return {mode:"ai",provider:"deepseek",model:"deepseek-chat",systemPrompt:"Process this item:",prompt:"{{item}}",code:"return item.toUpperCase();",template:"Item: {{item}}",separator:"\\n---\\n",maxItems:20,outputFormat:"joined"};
  if (t==="setvar")   return {varName:"myVar",passThrough:true};
  if (t==="getvar")   return {varName:"myVar",defaultValue:""};
  if (t==="scraper")  return {url:"{{input}}",mode:"text",useCorsProxy:true,maxChars:6000};
  if (t==="notify")   return {platform:"discord",webhookUrl:"",messageTemplate:"{{input}}",customBody:'{"message":"{{input}}"}'};
  if (t==="note")     return {text:"Note...",color:N.a3};
  return {};
}

function mkNode(type, x, y, extra={}) {
  return {id:uid(), type, x, y, label:ND[type]?.label||type, config:defCfg(type), status:"idle", result:null, error:null, ...extra};
}

const dn1 = mkNode("trigger",60,200,{label:"Start"});
const dn2 = mkNode("ai",360,200,{label:"Research Agent",config:{provider:"deepseek",model:"deepseek-chat",systemPrompt:"You are a research assistant. Write a detailed summary.",temperature:0.7,maxTokens:1024,inputVar:"{{input}}"}});
const dn3 = mkNode("text",660,80,{label:"Content Brief",config:{template:"Convert to brief:\n\n{{input}}\n\nTitle, Audience, Key Points (5)"}});
const dn4 = mkNode("code",660,320,{label:"Extract Stats",config:{code:"return JSON.stringify({lines:input.split('\\n').length,chars:input.length},null,2);"}});
const dn5 = mkNode("output",960,200,{label:"Result"});
const DEFAULT_NODES = [dn1,dn2,dn3,dn4,dn5];
const DEFAULT_EDGES = [
  {id:"e1",fn:dn1.id,fp:0,tn:dn2.id,tp:0},
  {id:"e2",fn:dn2.id,fp:0,tn:dn3.id,tp:0},
  {id:"e3",fn:dn2.id,fp:0,tn:dn4.id,tp:0},
  {id:"e4",fn:dn3.id,fp:0,tn:dn5.id,tp:0},
];

async function callAI(cfg, key, msg, backendUrl, backendMode) {
  if (backendMode && backendUrl) {
    const r = await fetch(`${backendUrl}/execute`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({provider:cfg.provider,model:cfg.model,system_prompt:cfg.systemPrompt||"",user_message:msg,temperature:cfg.temperature,max_tokens:cfg.maxTokens})});
    if (!r.ok) throw new Error(`Backend HTTP ${r.status}`);
    return (await r.json()).result;
  }
  const p = PROVS[cfg.provider];
  if (!p) throw new Error("Unknown provider");
  if (p.native) {
    const r = await fetch(p.ep, {method:"POST", headers:{"Content-Type":"application/json","x-api-key":key,"anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"}, body:JSON.stringify({model:cfg.model,max_tokens:cfg.maxTokens,system:cfg.systemPrompt,messages:[{role:"user",content:msg}]})});
    const d = await r.json(); return d.content?.[0]?.text || JSON.stringify(d);
  }
  const hdrs = {"Content-Type":"application/json","Authorization":`Bearer ${key||"ollama"}`};
  if (cfg.provider==="openrouter"){hdrs["HTTP-Referer"]="https://hermes.local";hdrs["X-Title"]="Hermes";}
  const r = await fetch(p.ep, {method:"POST", headers:hdrs, body:JSON.stringify({model:cfg.model,max_tokens:cfg.maxTokens,temperature:cfg.temperature,stream:false,messages:[...(cfg.systemPrompt?[{role:"system",content:cfg.systemPrompt}]:[]),{role:"user",content:msg}]})});
  if (!r.ok) throw new Error(`${cfg.provider} HTTP ${r.status}`);
  const d = await r.json(); return d.choices?.[0]?.message?.content || JSON.stringify(d);
}

async function execNode(node, input, keys, vars, backendUrl, backendMode) {
  const v = vars || {};
  switch (node.type) {
    case "trigger":  return subVars(node.config.triggerData||"Workflow started.", v);
    case "schedule": return `Triggered at: ${new Date().toLocaleString()}`;
    case "file":     return node.config.content || "(empty)";
    case "note":     return input;
    case "setvar":   v[node.config.varName]=input; return node.config.passThrough!==false ? input : `Stored→${node.config.varName}`;
    case "getvar":   return v[node.config.varName] ?? subVars(node.config.defaultValue||"", v);
    case "output":   return input;
    case "webhook": {
      try {
        const pl = JSON.parse(node.config.simPayload||"{}");
        if (node.config.extractField?.trim()) { let val=pl; for (const f of node.config.extractField.split(".")) val=val?.[f]; return String(val ?? JSON.stringify(pl, null, 2)); }
        return JSON.stringify(pl, null, 2);
      } catch { return node.config.simPayload||""; }
    }
    case "subflow": {
      if (!node.config.workflowJson) throw new Error("No workflow loaded");
      const {nodes:sn, edges:se} = JSON.parse(node.config.workflowJson);
      return await runSubWorkflow(sn, se, input, keys, node.config.passVars ? v : {});
    }
    case "ai": {
      const k = keys[node.config.provider];
      if (!k && !PROVS[node.config.provider]?.noKey) throw new Error(`No API key for ${node.config.provider}`);
      return await callAI({...node.config, systemPrompt:subVars(node.config.systemPrompt||"",v)}, k, subVars((node.config.inputVar||"{{input}}").replace(/\{\{input\}\}/g, input), v), backendUrl, backendMode);
    }
    case "http": { const r = await fetch(subVars(node.config.url, v), {method:node.config.method||"GET"}); return await r.text(); }
    case "scraper": {
      const rawUrl = subVars((node.config.url||input).replace(/\{\{input\}\}/g, input), v);
      const fetchUrl = node.config.useCorsProxy ? `https://corsproxy.io/?${encodeURIComponent(rawUrl)}` : rawUrl;
      const r = await fetch(fetchUrl); if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const html = await r.text();
      if (node.config.mode==="json") { try { return JSON.stringify(JSON.parse(html),null,2); } catch { return html; } }
      if (node.config.mode==="html") return html.substring(0, 8000);
      if (node.config.mode==="links") { const links=[...html.matchAll(/href="(https?[^"]+)"/g)].map(m=>m[1]).filter((x,i,a)=>a.indexOf(x)===i); return links.slice(0,50).join("\n"); }
      return html.replace(/<script[\s\S]*?<\/script>/gi,"").replace(/<style[\s\S]*?<\/style>/gi,"").replace(/<[^>]+>/g," ").replace(/&[a-z#0-9]+;/gi," ").replace(/\s+/g," ").trim().substring(0, node.config.maxChars||6000);
    }
    case "code": { try { const fn = new Function("input","vars", node.config.code); return String(fn(input,v)||""); } catch(e) { throw new Error(`JS: ${e.message}`); } }
    case "text":   return subVars((node.config.template||"{{input}}").replace(/\{\{input\}\}/g, input), v);
    case "split": { const d = (node.config.delimiter||"\\n\\n").replace(/\\n/g,"\n"); return input.split(d).filter(p=>p.trim()).slice(0,node.config.maxParts||10).join("\n|||SPLIT|||\n"); }
    case "merge":  return input;
    case "filter": { try { const f = new Function("input","vars",`return !!(${node.config.condition})`); return f(input,v) ? input : "__filtered_out__"; } catch { return input; } }
    case "loop": {
      let items = input.includes("|||SPLIT|||") ? input.split("|||SPLIT|||").map(s=>s.trim()).filter(Boolean) : [];
      if (!items.length) { try { const p=JSON.parse(input); items=Array.isArray(p)?p:[input]; } catch { items=[input]; } }
      items = items.slice(0, node.config.maxItems||20);
      const parts = [];
      for (let i=0; i<items.length; i++) {
        const item = items[i];
        if (node.config.mode==="code") { try { const fn=new Function("item","index","vars",node.config.code); parts.push(String(fn(item,i,v)||"")); } catch(e) { parts.push(`[Error:${e.message}]`); } }
        else if (node.config.mode==="template") parts.push(subVars(node.config.template.replace(/\{\{item\}\}/g,item), v));
        else { const k=keys[node.config.provider]; parts.push(await callAI({...node.config,systemPrompt:subVars(node.config.systemPrompt||"",v)}, k, subVars((node.config.prompt||"{{item}}").replace(/\{\{item\}\}/g,item), v), backendUrl, backendMode)); }
      }
      const sep = (node.config.separator||"\\n---\\n").replace(/\\n/g,"\n");
      return node.config.outputFormat==="array" ? JSON.stringify(parts) : parts.join(sep);
    }
    case "notify": {
      const url = node.config.webhookUrl; if (!url?.startsWith("http")) throw new Error("Invalid webhook URL");
      const msg = subVars((node.config.messageTemplate||"{{input}}").replace(/\{\{input\}\}/g,input), v);
      let body;
      if (node.config.platform==="discord") body={content:msg,username:"Hermes"};
      else if (node.config.platform==="slack") body={text:msg};
      else { try { body=JSON.parse(subVars(node.config.customBody||'{"message":"{{input}}"}',v).replace(/\{\{input\}\}/g,msg)); } catch { body={message:msg}; } }
      const r = await fetch(url, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
      return `Sent to ${node.config.platform} (${r.status})`;
    }
    default: return input;
  }
}

async function runSubWorkflow(sn, se, triggerInput, keys, parentVars) {
  const results = {}, lv = {...parentVars};
  const inE = {}; se.forEach(e => { (inE[e.tn]=inE[e.tn]||[]).push(e); });
  const order = topoSort(sn, se);
  for (const nid of order) {
    const node = sn.find(n=>n.id===nid); if (!node) continue;
    const deps = inE[nid]||[];
    const inp = deps.map(e=>results[e.fn]||"").join("\n---\n");
    const isRoot = ND[node.type]?.ins===0;
    const n2 = isRoot ? {...node,config:{...node.config,triggerData:triggerInput,simPayload:triggerInput}} : node;
    try { results[nid]=await execNode(n2,inp,keys,lv); } catch(e) { results[nid]=`[Error:${e.message}]`; }
  }
  const outNode = sn.find(n=>n.type==="output");
  if (outNode && results[outNode.id]) return results[outNode.id];
  return results[order[order.length-1]] || "";
}

function detectType(s) {
  if (!s?.trim()) return {icon:"∅",label:"Empty"};
  if (s.includes("|||SPLIT|||")) { const n=s.split("|||SPLIT|||").filter(p=>p.trim()).length; return {icon:"⑂",label:`Split · ${n} items`}; }
  try { const j=JSON.parse(s); return Array.isArray(j) ? {icon:"[]",label:`JSON array · ${j.length} items`} : {icon:"{}",label:`JSON · ${Object.keys(j).length} keys`}; } catch {}
  if (/\b(def |function |import |const |class )\b/.test(s)&&s.split("\n").length>3) return {icon:"</>",label:"Code"};
  if (s.includes("**")||s.includes("##")||s.includes("- ")) return {icon:"¶",label:"Markdown"};
  return {icon:"T",label:"Plain text"};
}

function genPy(nodes, edges) {
  const order = topoSort(nodes, edges);
  const inE = {}; edges.forEach(e => { (inE[e.tn]=inE[e.tn]||[]).push(e); });
  const vn = id => `r_${id.replace(/[^a-z0-9]/gi,"_")}`;
  const dep = nid => { const d=inE[nid]||[]; if(!d.length)return'""'; if(d.length===1)return`results["${vn(d[0].fn)}"]`; return`"\\n---\\n".join([${d.map(e=>`results["${vn(e.fn)}"]`).join(",")}])`; };
  const lines = ["#!/usr/bin/env python3","# Hermes Workflow Export — pip install httpx requests","import asyncio,os,json,requests,httpx","","async def run_workflow():","    results={}","    vars_store={}",""];
  for (const nid of order) {
    const node = nodes.find(n=>n.id===nid); if (!node) continue;
    const v = vn(nid), d = dep(nid);
    lines.push(`    # -- ${node.label} (${node.type})`);
    if (["trigger","webhook","schedule"].includes(node.type)) lines.push(`    results["${v}"] = "${(node.config.triggerData||node.config.simPayload||"").replace(/"/g,'\\"').substring(0,200)}"`);
    else if (node.type==="ai") lines.push(`    results["${v}"] = await hermes_complete("${node.config.provider}","${node.config.model}",${d})`);
    else if (node.type==="code") lines.push(`    # JS: ${(node.config.code||"").split("\n")[0]}\n    results["${v}"] = ${d}  # TODO`);
    else if (node.type==="http") lines.push(`    results["${v}"] = requests.${(node.config.method||"GET").toLowerCase()}("${node.config.url}").text`);
    else if (node.type==="output") lines.push(`    print(${d})\n    results["${v}"] = ${d}`);
    else if (node.type==="setvar") lines.push(`    vars_store["${node.config.varName}"] = ${d}\n    results["${v}"] = ${d}`);
    else if (node.type==="getvar") lines.push(`    results["${v}"] = vars_store.get("${node.config.varName}","")`);
    else if (node.type==="note") lines.push(`    # NOTE: ${(node.config.text||"").replace(/\n/g," ")}`);
    else lines.push(`    results["${v}"] = ${d}`);
    lines.push("");
  }
  lines.push("","if __name__ == '__main__':","    asyncio.run(run_workflow())","");
  return lines.join("\n");
}

// ── Shared styles ──────────────────────────────────────────────────────────
const IS = {width:"100%",background:N.n1,border:`1px solid ${N.n3}`,borderRadius:6,padding:"7px 10px",color:N.n5,fontSize:11,fontFamily:"JetBrains Mono,monospace"};
const SL = {width:"100%",background:N.n1,border:`1px solid ${N.n3}`,borderRadius:6,padding:"7px 10px",color:N.n5,fontSize:11,fontFamily:"JetBrains Mono,monospace",cursor:"pointer"};
const TS = {width:"100%",background:N.n1,border:`1px solid ${N.n3}`,borderRadius:6,padding:"7px 10px",color:N.n5,fontSize:11,fontFamily:"JetBrains Mono,monospace",resize:"vertical",lineHeight:1.65};
const BS = (on,c) => ({background:on?`${c}15`:"transparent",border:`1px solid ${on?`${c}50`:N.n2}`,borderRadius:6,padding:"6px 12px",color:on?c:N.n3,fontSize:9,letterSpacing:1.5,cursor:on?"pointer":"default",opacity:on?1:.4,transition:"all .15s",fontFamily:"JetBrains Mono,monospace"});

function Lbl({ children }) {
  return (
    <div style={{fontSize:9,color:N.f3,letterSpacing:2,marginBottom:5,fontWeight:600}}>{children}</div>
  );
}

function Fld({ label, children }) {
  return (
    <div style={{marginBottom:14}}>
      <Lbl>{label}</Lbl>
      {children}
    </div>
  );
}

function SFld({ label, value, min, max, step, onChange, color }) {
  return (
    <Fld label={`${label}: ${value}`}>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{width:"100%",accentColor:color,height:3,cursor:"pointer"}} />
    </Fld>
  );
}

// ── NodeCard ────────────────────────────────────────────────────────────────
function NodeCard({ node, selected, multiSel, connecting, stepStatus, onSelect, onDragStart, onStartConn, onEndConn, onCtxMenu }) {
  const def = ND[node.type] || {color:N.f3,icon:"?",ins:0,outs:0};
  const color = def.color;
  const running = node.status==="running", done = node.status==="done", err = node.status==="error";
  const sc = running?N.a3:done?N.a4:err?N.a1:"transparent";
  const stepBorder = stepStatus==="next"?N.a3:stepStatus==="done"?N.a4:stepStatus==="queued"?N.f3:null;
  const stepOpacity = stepStatus==="waiting" ? 0.35 : 1;
  const borderColor = stepBorder || (multiSel ? "#ff930060" : selected ? N.f3 : N.n2);

  return (
    <div
      onMouseDown={e => { e.stopPropagation(); if (e.button===0) { onDragStart(e,node.id); onSelect(node.id); } }}
      onContextMenu={e => { e.preventDefault(); e.stopPropagation(); if (onCtxMenu) onCtxMenu(e,node.id); }}
      style={{position:"absolute",left:node.x,top:node.y,width:NW,background:N.n1,borderRadius:10,
        opacity:stepOpacity, border:`1.5px solid ${borderColor}`, borderTop:`3px solid ${running?N.a3:stepBorder||color}`,
        boxShadow:stepStatus==="next"?`0 0 20px ${N.a3}60,0 4px 16px rgba(0,0,0,.4)`:selected?`0 8px 24px rgba(0,0,0,.45)`:`0 4px 12px rgba(0,0,0,.35)`,
        cursor:"grab",userSelect:"none",transition:"all .2s",zIndex:selected?10:1}}>
      {running && (
        <div style={{position:"absolute",top:0,left:0,right:0,height:3,background:`linear-gradient(90deg,transparent,${N.a3},transparent)`,borderRadius:"8px 8px 0 0",animation:"sweep 1.6s infinite"}} />
      )}
      {Array.from({length:def.ins}).map((_,i) => {
        const p = inP(node,i);
        return (
          <div key={i}
            onMouseUp={e => { e.stopPropagation(); if (connecting) onEndConn(node.id,i); }}
            style={{position:"absolute",left:-PR,top:p.y-node.y-PR,width:PR*2,height:PR*2,borderRadius:"50%",
              background:connecting?`${N.f2}30`:N.n0,border:`2.5px solid ${connecting?N.f2:N.n3}`,
              cursor:"crosshair",zIndex:20,transition:"all .15s",boxShadow:connecting?`0 0 8px ${N.f2}60`:"none"}} />
        );
      })}
      <div style={{padding:"0 14px",height:NH,display:"flex",alignItems:"center",gap:10}}>
        <span style={{fontSize:15,color,flexShrink:0}}>{def.icon}</span>
        <div style={{flex:1,minWidth:0}}>
          <div style={{fontSize:11.5,fontWeight:600,color:N.n5,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{node.label}</div>
          <div style={{fontSize:9,color:N.f3,letterSpacing:1,marginTop:1}}>{node.type}</div>
        </div>
        <div style={{width:8,height:8,borderRadius:"50%",background:sc,flexShrink:0,
          boxShadow:running?`0 0 8px ${N.a3}`:done?`0 0 6px ${N.a4}`:err?`0 0 6px ${N.a1}`:"none",transition:"all .3s"}} />
      </div>
      {(done||err) && (
        <div style={{borderTop:`1px solid ${N.n2}`,padding:"7px 12px 9px"}}>
          <div style={{fontSize:10,color:err?`${N.a1}cc`:`${N.f1}cc`,fontFamily:"JetBrains Mono,monospace",lineHeight:1.5,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>
            {err ? `⚠ ${node.error}` : `✓ ${(node.result||"").substring(0,60)}${(node.result||"").length>60?"…":""}`}
          </div>
        </div>
      )}
      {Array.from({length:def.outs}).map((_,i) => {
        const p = outP(node,i);
        return (
          <div key={i}
            onMouseDown={e => { e.stopPropagation(); onStartConn(node.id,i); }}
            onMouseEnter={e => { e.currentTarget.style.transform="scale(1.3)"; }}
            onMouseLeave={e => { e.currentTarget.style.transform="scale(1)"; }}
            style={{position:"absolute",right:-PR,top:p.y-node.y-PR,width:PR*2,height:PR*2,borderRadius:"50%",
              background:color,border:`2.5px solid ${N.n0}`,cursor:"crosshair",zIndex:20,
              boxShadow:`0 0 8px ${color}70`,transition:"transform .15s"}} />
        );
      })}
    </div>
  );
}

// ── ConfigPanel ─────────────────────────────────────────────────────────────
function ConfigPanel({ node, apiKeys, onChange }) {
  if (!node) {
    return (
      <div style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",height:"100%",gap:12,opacity:.4}}>
        <div style={{fontSize:32,color:N.n3}}>◈</div>
        <div style={{fontSize:10,color:N.f3,letterSpacing:2}}>SELECT A NODE</div>
      </div>
    );
  }
  const def = ND[node.type] || {color:N.f3};
  const color = def.color;
  const upd = (k, val) => onChange({...node, config:{...node.config,[k]:val}});
  const prov = PROVS[node.config?.provider];

  const providerSelect = (
    <Fld label="PROVIDER">
      <select value={node.config.provider}
        onChange={e => onChange({...node,config:{...node.config,provider:e.target.value,model:PROVS[e.target.value]?.def||""}})}
        style={SL}>
        {Object.entries(PROVS).map(([k,v]) => <option key={k} value={k}>{v.name}</option>)}
      </select>
    </Fld>
  );
  const modelSelect = (
    <Fld label="MODEL">
      <select value={node.config.model} onChange={e => upd("model",e.target.value)} style={SL}>
        {(prov?.models||[]).map(m => <option key={m} value={m}>{m.split("/").pop()}</option>)}
      </select>
    </Fld>
  );

  return (
    <div style={{padding:"16px 18px",overflowY:"auto",flex:1}}>
      <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:18,paddingBottom:14,borderBottom:`1px solid ${N.n2}`}}>
        <div style={{width:36,height:36,borderRadius:8,background:N.n2,display:"flex",alignItems:"center",justifyContent:"center",fontSize:17,color,flexShrink:0}}>{def.icon||"?"}</div>
        <div style={{flex:1}}>
          <input value={node.label} onChange={e => onChange({...node,label:e.target.value})}
            style={{background:"none",border:"none",borderBottom:`1px solid ${N.n3}`,color:N.n5,fontSize:13,fontWeight:600,fontFamily:"JetBrains Mono,monospace",width:"100%",padding:"2px 0"}} />
          <div style={{fontSize:9,color,letterSpacing:2,marginTop:3}}>{node.type.toUpperCase()}</div>
        </div>
      </div>

      {node.type==="trigger" && <Fld label="INITIAL DATA"><textarea value={node.config.triggerData||""} onChange={e=>upd("triggerData",e.target.value)} rows={4} style={TS} /></Fld>}

      {node.type==="webhook" && <>
        <Fld label="SIMULATED PAYLOAD"><textarea value={node.config.simPayload} onChange={e=>upd("simPayload",e.target.value)} rows={4} style={{...TS,color:N.a5,fontSize:11}} placeholder='{"event":"trigger"}' /></Fld>
        <Fld label="EXTRACT FIELD (dot notation)"><input value={node.config.extractField} onChange={e=>upd("extractField",e.target.value)} style={IS} placeholder="data.message" /></Fld>
      </>}

      {node.type==="ai" && <>
        {providerSelect}{modelSelect}
        <div style={{fontSize:9,color:apiKeys[node.config.provider]?N.a4:N.a1,marginBottom:14}}>{apiKeys[node.config.provider]?"✓ Key set":"⚠ No key — set in KEYS"}</div>
        <Fld label="SYSTEM PROMPT"><textarea value={node.config.systemPrompt} onChange={e=>upd("systemPrompt",e.target.value)} rows={5} style={TS} /></Fld>
        <Fld label="INPUT VAR"><input value={node.config.inputVar||"{{input}}"} onChange={e=>upd("inputVar",e.target.value)} style={IS} /></Fld>
        <SFld label="Temperature" value={node.config.temperature} min={0} max={2} step={.05} onChange={v=>upd("temperature",v)} color={color} />
        <SFld label="Max Tokens" value={node.config.maxTokens} min={128} max={8192} step={128} onChange={v=>upd("maxTokens",v)} color={color} />
      </>}

      {node.type==="http" && <>
        <Fld label="URL"><input value={node.config.url} onChange={e=>upd("url",e.target.value)} style={IS} /></Fld>
        <Fld label="METHOD"><select value={node.config.method} onChange={e=>upd("method",e.target.value)} style={SL}>{["GET","POST","PUT","DELETE","PATCH"].map(m=><option key={m}>{m}</option>)}</select></Fld>
      </>}

      {node.type==="scraper" && <>
        <Fld label="URL ({{input}})"><input value={node.config.url} onChange={e=>upd("url",e.target.value)} style={IS} placeholder="https:// or {{input}}" /></Fld>
        <Fld label="MODE">
          <div style={{display:"flex",gap:4}}>
            {["text","html","links","json"].map(m => (
              <button key={m} onClick={()=>upd("mode",m)} style={{flex:1,background:node.config.mode===m?`${color}25`:"transparent",border:`1px solid ${node.config.mode===m?color:N.n3}`,borderRadius:5,padding:"5px 0",color:node.config.mode===m?color:N.n3,fontSize:9,cursor:"pointer"}}>{m.toUpperCase()}</button>
            ))}
          </div>
        </Fld>
      </>}

      {node.type==="subflow" && <>
        <Fld label="WORKFLOW FILE">
          <label style={{display:"block",background:N.n2,border:`1.5px dashed ${node.config.workflowJson?N.f3:N.n3}`,borderRadius:6,padding:"12px",fontSize:10,textAlign:"center",cursor:"pointer",color:node.config.workflowJson?N.f3:N.n3}}>
            {node.config.workflowJson ? `✓ ${node.config.workflowName}` : "⊛ Load workflow.json"}
            <input type="file" accept=".json" style={{display:"none"}} onChange={e=>{
              const f=e.target.files?.[0]; if(!f)return;
              const r=new FileReader();
              r.onload=ev=>{ try{const p=JSON.parse(ev.target.result); onChange({...node,label:f.name.replace(".json",""),config:{...node.config,workflowJson:ev.target.result,workflowName:`${f.name} (${p.nodes?.length||0} nodes)`,passVars:node.config.passVars}});}catch{alert("Invalid JSON");} };
              r.readAsText(f); e.target.value="";
            }} />
          </label>
        </Fld>
      </>}

      {node.type==="code" && <Fld label="JAVASCRIPT"><textarea value={node.config.code} onChange={e=>upd("code",e.target.value)} rows={9} style={{...TS,color:N.a5,fontSize:11}} /></Fld>}
      {node.type==="text" && <Fld label="TEMPLATE ({{input}})"><textarea value={node.config.template} onChange={e=>upd("template",e.target.value)} rows={7} style={TS} /></Fld>}

      {node.type==="file" && <>
        <Fld label="FILE">
          <label style={{display:"block",background:N.n2,border:`1.5px dashed ${N.n3}`,borderRadius:6,padding:"12px",fontSize:10,color:N.n3,cursor:"pointer",textAlign:"center"}}>
            {node.config.filename ? `✓ ${node.config.filename}` : "⊞ Load file"}
            <input type="file" style={{display:"none"}} accept=".txt,.md,.py,.js,.ts,.json,.html,.css" onChange={e=>{ const f=e.target.files?.[0]; if(!f)return; const r=new FileReader(); r.onload=ev=>onChange({...node,label:f.name,config:{...node.config,content:ev.target.result,filename:f.name}}); r.readAsText(f); }} />
          </label>
        </Fld>
        <Fld label="OR PASTE"><textarea value={node.config.content} onChange={e=>upd("content",e.target.value)} rows={5} style={TS} placeholder="paste text…" /></Fld>
      </>}

      {node.type==="split" && <><Fld label="DELIMITER"><input value={node.config.delimiter} onChange={e=>upd("delimiter",e.target.value)} style={IS} placeholder="\n\n" /></Fld><SFld label="Max Parts" value={node.config.maxParts||10} min={2} max={50} step={1} onChange={v=>upd("maxParts",v)} color={color} /></>}
      {node.type==="filter" && <Fld label="CONDITION (JS)"><input value={node.config.condition} onChange={e=>upd("condition",e.target.value)} style={IS} placeholder="input.length > 0" /></Fld>}
      {node.type==="merge" && <Fld label="SEPARATOR"><input value={node.config.separator} onChange={e=>upd("separator",e.target.value)} style={IS} placeholder="\n---\n" /></Fld>}

      {node.type==="loop" && <>
        <Fld label="MODE">
          <div style={{display:"flex",gap:4}}>
            {["ai","code","template"].map(m => (
              <button key={m} onClick={()=>upd("mode",m)} style={{flex:1,background:node.config.mode===m?`${color}25`:"transparent",border:`1px solid ${node.config.mode===m?color:N.n3}`,borderRadius:5,padding:"5px 0",color:node.config.mode===m?color:N.n3,fontSize:9,cursor:"pointer"}}>{m.toUpperCase()}</button>
            ))}
          </div>
        </Fld>
        {node.config.mode==="ai" && <>{providerSelect}<Fld label="ITEM PROMPT ({{item}})"><textarea value={node.config.prompt} onChange={e=>upd("prompt",e.target.value)} rows={2} style={TS} /></Fld></>}
        {node.config.mode==="code" && <Fld label="JS (item, index, vars)"><textarea value={node.config.code} onChange={e=>upd("code",e.target.value)} rows={5} style={{...TS,color:N.a5,fontSize:11}} /></Fld>}
        {node.config.mode==="template" && <Fld label="TEMPLATE ({{item}})"><textarea value={node.config.template} onChange={e=>upd("template",e.target.value)} rows={4} style={TS} /></Fld>}
        <SFld label="Max Items" value={node.config.maxItems||20} min={1} max={100} step={1} onChange={v=>upd("maxItems",v)} color={color} />
      </>}

      {node.type==="setvar" && <Fld label="VARIABLE NAME"><input value={node.config.varName} onChange={e=>upd("varName",e.target.value)} style={IS} placeholder="myVar" /></Fld>}
      {node.type==="getvar" && <><Fld label="VARIABLE NAME"><input value={node.config.varName} onChange={e=>upd("varName",e.target.value)} style={IS} placeholder="myVar" /></Fld><Fld label="DEFAULT VALUE"><input value={node.config.defaultValue||""} onChange={e=>upd("defaultValue",e.target.value)} style={IS} /></Fld></>}

      {node.type==="notify" && <>
        <Fld label="PLATFORM">
          <div style={{display:"flex",gap:4}}>
            {["discord","slack","webhook"].map(p => (
              <button key={p} onClick={()=>upd("platform",p)} style={{flex:1,background:node.config.platform===p?`${color}25`:"transparent",border:`1px solid ${node.config.platform===p?color:N.n3}`,borderRadius:5,padding:"5px 0",color:node.config.platform===p?color:N.n3,fontSize:9,cursor:"pointer"}}>{p.toUpperCase()}</button>
            ))}
          </div>
        </Fld>
        <Fld label="WEBHOOK URL"><input value={node.config.webhookUrl} onChange={e=>upd("webhookUrl",e.target.value)} style={IS} placeholder="https://…" /></Fld>
        <Fld label="MESSAGE TEMPLATE"><textarea value={node.config.messageTemplate} onChange={e=>upd("messageTemplate",e.target.value)} rows={3} style={TS} /></Fld>
      </>}

      {node.type==="note" && <>
        <Fld label="TEXT"><textarea value={node.config.text} onChange={e=>upd("text",e.target.value)} rows={6} style={TS} /></Fld>
        <Fld label="COLOR">
          <div style={{display:"flex",gap:8}}>
            {[N.a3,N.a4,N.f2,N.a5,N.a1,N.f1].map(c => (
              <div key={c} onClick={()=>upd("color",c)} style={{width:20,height:20,borderRadius:"50%",background:c,cursor:"pointer",border:`2px solid ${node.config.color===c?"#fff":"transparent"}`}} />
            ))}
          </div>
        </Fld>
      </>}

      {node.result && (
        <div style={{marginTop:4,paddingTop:14,borderTop:`1px solid ${N.n2}`}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
            <Lbl>OUTPUT</Lbl>
            <button onClick={()=>navigator.clipboard.writeText(node.result)} style={{background:"none",border:`1px solid ${N.n3}`,borderRadius:4,padding:"2px 8px",color:N.n3,fontSize:9,cursor:"pointer"}}>⎘ COPY</button>
          </div>
          <div style={{background:N.n0,border:`1px solid ${N.n2}`,borderRadius:6,padding:"10px 12px",fontSize:10.5,color:N.f1,fontFamily:"JetBrains Mono,monospace",whiteSpace:"pre-wrap",maxHeight:200,overflowY:"auto",lineHeight:1.65}}>{node.result}</div>
        </div>
      )}
    </div>
  );
}

// ── EdgeInspector ───────────────────────────────────────────────────────────
function EdgeInspector({ edge, nodes, edgeData, onDelete, onClose }) {
  const [copied, setCopied] = useState(false);
  if (!edge) {
    return (
      <div style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",height:"100%",gap:12,opacity:.4}}>
        <div style={{fontSize:28,color:N.n3}}>—</div>
        <div style={{fontSize:10,color:N.f3,letterSpacing:2}}>CLICK AN EDGE</div>
      </div>
    );
  }
  const data = edgeData[edge.id] || "";
  const dt = detectType(data);
  const fromNode = nodes.find(n=>n.id===edge.fn);
  const toNode = nodes.find(n=>n.id===edge.tn);
  const wordCount = data.split(/\s+/).filter(Boolean).length;
  const lineCount = data.split("\n").length;
  const tokenEst = Math.ceil(data.length/4);
  const copy = () => { navigator.clipboard.writeText(data); setCopied(true); setTimeout(()=>setCopied(false),1500); };
  const preview = () => {
    if (dt.icon==="{}"||dt.icon==="[]") { try { return JSON.stringify(JSON.parse(data),null,2); } catch {} }
    if (dt.icon==="⑂") return data.split("|||SPLIT|||").map((p,i)=>`[${i+1}] ${p.trim()}`).join("\n\n");
    return data;
  };
  return (
    <div style={{padding:"16px 18px",overflowY:"auto",flex:1,display:"flex",flexDirection:"column"}}>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:16,paddingBottom:14,borderBottom:`1px solid ${N.n2}`}}>
        <div style={{flex:1}}>
          <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:3}}>
            <span style={{fontSize:11,color:ND[fromNode?.type]?.color||N.f3,fontWeight:600}}>{fromNode?.label||"?"}</span>
            <span style={{fontSize:13,color:N.n3}}>→</span>
            <span style={{fontSize:11,color:ND[toNode?.type]?.color||N.f3,fontWeight:600}}>{toNode?.label||"?"}</span>
          </div>
          <span style={{fontSize:9,color:N.n3,background:N.n2,padding:"2px 7px",borderRadius:8}}>{dt.icon} {dt.label}</span>
        </div>
        <button onClick={onClose} style={{background:"none",border:"none",color:N.n3,cursor:"pointer",fontSize:16,lineHeight:1}}>×</button>
      </div>
      {data && (
        <div style={{display:"flex",gap:8,flexWrap:"wrap",marginBottom:14}}>
          {[{l:"Words",v:wordCount},{l:"Lines",v:lineCount},{l:"~Tokens",v:tokenEst},{l:"Chars",v:data.length}].map(s => (
            <div key={s.l} style={{background:N.n0,border:`1px solid ${N.n2}`,borderRadius:6,padding:"5px 10px",textAlign:"center",flex:1,minWidth:60}}>
              <div style={{fontSize:11,color:N.f3,fontWeight:600}}>{s.v.toLocaleString()}</div>
              <div style={{fontSize:8,color:N.n3,letterSpacing:1}}>{s.l}</div>
            </div>
          ))}
        </div>
      )}
      {data ? (
        <>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
            <div style={{fontSize:9,color:N.n3,letterSpacing:2}}>DATA</div>
            <button onClick={copy} style={{background:"none",border:`1px solid ${copied?N.a4:N.n3}`,borderRadius:4,padding:"2px 8px",color:copied?N.a4:N.n3,fontSize:9,cursor:"pointer"}}>{copied?"✓ Copied":"⎘ Copy"}</button>
          </div>
          <div style={{background:N.n0,border:`1px solid ${N.n2}`,borderRadius:6,padding:"10px 12px",fontSize:10.5,color:N.f1,fontFamily:"JetBrains Mono,monospace",whiteSpace:"pre-wrap",flex:1,overflowY:"auto",lineHeight:1.65,minHeight:100,maxHeight:320}}>{preview()}</div>
        </>
      ) : (
        <div style={{textAlign:"center",padding:"24px 0",color:N.n3,fontSize:10}}>No data yet — run the workflow first</div>
      )}
      <button onClick={onDelete} style={{marginTop:14,background:`${N.a1}10`,border:`1px solid ${N.a1}40`,borderRadius:6,padding:"7px",color:N.a1,fontSize:9,cursor:"pointer",letterSpacing:1}}>✕ Delete this edge</button>
    </div>
  );
}

// ── Minimap ─────────────────────────────────────────────────────────────────
function Minimap({ nodes, edges, pan, zoom, containerRef, onPanTo }) {
  const MW=200, MH=130;
  if (!nodes.length) return null;
  const xs = nodes.flatMap(n => [n.x, n.x+NW]);
  const ys = nodes.flatMap(n => [n.y, n.y+NH]);
  const bxMin=Math.min(...xs)-60, bxMax=Math.max(...xs)+60;
  const byMin=Math.min(...ys)-60, byMax=Math.max(...ys)+60;
  const bw=bxMax-bxMin, bh=byMax-byMin;
  if (bw<=0||bh<=0) return null;
  const sc = Math.min((MW-4)/bw, (MH-4)/bh);
  const tx = x => (x-bxMin)*sc+2;
  const ty = y => (y-byMin)*sc+2;
  const cw = containerRef.current?.clientWidth||800;
  const ch = containerRef.current?.clientHeight||600;
  const vx=-pan.x/zoom, vy=-pan.y/zoom, vw=cw/zoom, vh=ch/zoom;
  const handleClick = e => {
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = (e.clientX-rect.left-2)/sc+bxMin;
    const cy = (e.clientY-rect.top-2)/sc+byMin;
    onPanTo(-cx*zoom+cw/2, -cy*zoom+ch/2);
  };
  return (
    <div style={{position:"absolute",bottom:40,right:14,zIndex:50,background:`${N.n1}f0`,border:`1px solid ${N.n2}`,borderRadius:8,width:MW,height:MH,overflow:"hidden",cursor:"crosshair",boxShadow:"0 4px 16px rgba(0,0,0,.4)"}} onClick={handleClick}>
      <svg width={MW} height={MH} style={{display:"block"}}>
        {edges.map(e => {
          const fn=nodes.find(n=>n.id===e.fn), tn=nodes.find(n=>n.id===e.tn);
          if (!fn||!tn) return null;
          const a=outP(fn,e.fp), b=inP(tn,e.tp);
          return <line key={e.id} x1={tx(a.x)} y1={ty(a.y)} x2={tx(b.x)} y2={ty(b.y)} stroke={`${ND[fn.type]?.color||N.f3}50`} strokeWidth={1} />;
        }).filter(Boolean)}
        {nodes.map(n => (
          <rect key={n.id} x={tx(n.x)} y={ty(n.y)} width={Math.max(NW*sc,4)} height={Math.max(NH*sc,3)} rx={2} fill={`${ND[n.type]?.color||N.f3}30`} stroke={ND[n.type]?.color||N.f3} strokeWidth={.5} opacity={.8} />
        ))}
        <rect x={Math.max(2,tx(vx))} y={Math.max(2,ty(vy))} width={Math.min(vw*sc,MW-4)} height={Math.min(vh*sc,MH-4)} rx={2} fill="none" stroke={N.f3} strokeWidth={1.5} strokeDasharray="3,2" opacity={.8} />
      </svg>
      <div style={{position:"absolute",top:3,left:6,fontSize:7,color:N.n3,letterSpacing:1,pointerEvents:"none"}}>MAP</div>
    </div>
  );
}

// ── Generate server.py ──────────────────────────────────────────────────────
function genServer() {
  const provList = Object.entries(PROVS).filter(([,v])=>!v.noKey).map(([k,v])=>
    `    "${k}": {"name":"${v.name}","ep":"${v.ep}","native":${v.native?'True':'False'}}`
  ).join(",\n");

  return `#!/usr/bin/env python3
"""
Hermes Backend Server — bridges Hermes Workflows GUI to hermes_client.py
pip install fastapi uvicorn httpx python-dotenv
Run: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""
import os, json, asyncio
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import httpx

load_dotenv()  # reads .env file

app = FastAPI(title="Hermes Backend", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Provider registry (mirrors GUI) ─────────────────────────────────────────
PROVIDERS = {
${provList}
}

# ── API Keys (set in .env or env vars) ───────────────────────────────────────
KEYS = {k: os.getenv(k.upper() + "_API_KEY", "") for k in PROVIDERS}
KEYS["ollama"] = "ollama"

# ── Models ───────────────────────────────────────────────────────────────────
class ExecReq(BaseModel):
    provider: str
    model: str
    system_prompt: str = ""
    user_message: str
    temperature: float = 0.7
    max_tokens: int = 1024

class WorkflowReq(BaseModel):
    nodes: list
    edges: list
    vars: dict = {}

# ── Core completion ──────────────────────────────────────────────────────────
async def hermes_complete(provider: str, model: str, user_message: str,
                           system: str = "", temperature: float = 0.7,
                           max_tokens: int = 1024) -> str:
    p = PROVIDERS.get(provider)
    if not p:
        raise ValueError(f"Unknown provider: {provider}")
    key = KEYS.get(provider, "")
    if not key and not provider == "ollama":
        raise ValueError(f"No API key for {provider}. Set {provider.upper()}_API_KEY in .env")

    if p.get("native"):  # Anthropic
        async with httpx.AsyncClient() as c:
            r = await c.post(p["ep"],
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": model, "max_tokens": max_tokens, "system": system,
                      "messages": [{"role": "user", "content": user_message}]}, timeout=120)
            r.raise_for_status()
            return r.json()["content"][0]["text"]

    messages = ([{"role":"system","content":system}] if system else []) + [{"role":"user","content":user_message}]
    async with httpx.AsyncClient() as c:
        r = await c.post(p["ep"],
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "providers": list(PROVIDERS.keys()),
            "keys_set": [k for k,v in KEYS.items() if v and v != "ollama" or k == "ollama"]}

@app.get("/providers")
def providers():
    return {k: {"name":v["name"], "has_key": bool(KEYS.get(k))} for k,v in PROVIDERS.items()}

@app.post("/execute")
async def execute(req: ExecReq):
    try:
        result = await hermes_complete(req.provider, req.model, req.user_message,
                                        req.system_prompt, req.temperature, req.max_tokens)
        return {"result": result, "provider": req.provider, "model": req.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/stream")
async def stream(ws: WebSocket):
    """Streaming endpoint — sends chunks as they arrive."""
    await ws.accept()
    try:
        data = await ws.receive_json()
        p = PROVIDERS.get(data["provider"])
        if not p:
            await ws.send_json({"error": f"Unknown provider: {data['provider']}"}); return
        key = KEYS.get(data["provider"], "")
        messages = ([{"role":"system","content":data.get("system_prompt","")}] if data.get("system_prompt") else [])
        messages += [{"role":"user","content":data["user_message"]}]

        async with httpx.AsyncClient() as c:
            async with c.stream("POST", p["ep"],
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": data["model"], "messages": messages, "stream": True,
                      "temperature": data.get("temperature", 0.7), "max_tokens": data.get("max_tokens", 1024)},
                timeout=120) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:].strip()
                        if chunk == "[DONE]": break
                        try:
                            delta = json.loads(chunk)["choices"][0]["delta"].get("content","")
                            if delta: await ws.send_text(delta)
                        except: pass
        await ws.send_text("[DONE]")
    except Exception as e:
        await ws.send_json({"error": str(e)})
    finally:
        await ws.close()

@app.post("/workflow")
async def run_workflow(req: WorkflowReq):
    """Run a full workflow server-side (for RecursiveLearner / AgentMesh integration)."""
    # TODO: import and use hermes_client.py RecursiveLearner + SkillRegistry here
    results = {}
    for node in req.nodes:
        if node["type"] == "ai":
            cfg = node.get("config", {})
            try:
                result = await hermes_complete(cfg.get("provider","deepseek"), cfg.get("model","deepseek-chat"),
                    cfg.get("inputVar","{{input}}").replace("{{input}}", ""), cfg.get("systemPrompt",""),
                    cfg.get("temperature", 0.7), cfg.get("maxTokens", 1024))
                results[node["id"]] = result
            except Exception as e:
                results[node["id"]] = f"[Error: {e}]"
    return {"results": results, "vars": req.vars}

if __name__ == "__main__":
    import uvicorn
    print("Starting Hermes Backend on http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
`;
}

// ── WorkflowBuilder ─────────────────────────────────────────────────────────
export default function WorkflowBuilder() {
  const [nodes,    setNodes]    = useState(DEFAULT_NODES);
  const [edges,    setEdges]    = useState(DEFAULT_EDGES);
  const [selId,    setSelId]    = useState(dn1.id);
  const [pan,      setPan]      = useState({x:40,y:20});
  const [conn,     setConn]     = useState(null);
  const [mpos,     setMpos]     = useState({x:0,y:0});
  const [zoom,     setZoom]     = useState(1);
  const [apiKeys,  setApiKeys]  = useState(Object.fromEntries(Object.keys(PROVS).map(k=>[k,k==="ollama"?"ollama":""])));
  const [backendUrl,   setBackendUrl]   = useState("http://localhost:8000");
  const [backendMode,  setBackendMode]  = useState(false);
  const [backendHealth,setBackendHealth]= useState("idle"); // idle | checking | ok | error
  const [running,  setRunning]  = useState(false);
  const [log,      setLog]      = useState([]);
  const [showKeys, setShowKeys] = useState(false);
  const [showLog,  setShowLog]  = useState(false);
  const [canUndo,  setCanUndo]  = useState(false);
  const [canRedo,  setCanRedo]  = useState(false);
  const [vars,     setVars]     = useState({});
  const [showVars, setShowVars] = useState(false);
  const [edgeData, setEdgeData] = useState({});
  const [hoverEdge,setHoverEdge]= useState(null);
  const [selEdge,  setSelEdge]  = useState(null);
  const [stepMode, setStepMode] = useState(false);
  const [stepQueue,setStepQueue]= useState([]);
  const [stepDone, setStepDone] = useState(new Set());
  const [stepRun,  setStepRun]  = useState(false);
  const [showExp,  setShowExp]  = useState(false);
  const [expCode,  setExpCode]  = useState("");
  const [multiSel, setMultiSel] = useState(new Set());
  const [selBox,   setSelBox]   = useState(null);
  const [ctxMenu,  setCtxMenu]  = useState(null);
  const [showMap,  setShowMap]  = useState(true);

  const drag = useRef(null);
  const cRef = useRef(null);
  const loadRef = useRef(null);
  const undoStack = useRef([]);
  const redoStack = useRef([]);
  const stepResultsRef = useRef({});
  const stepVarsRef = useRef({});

  const snapshot = useCallback((ns=nodes, es=edges) => {
    undoStack.current = [...undoStack.current.slice(-49), {nodes:JSON.parse(JSON.stringify(ns)), edges:JSON.parse(JSON.stringify(es))}];
    redoStack.current = [];
    setCanUndo(true); setCanRedo(false);
  }, [nodes, edges]);

  const undo = useCallback(() => {
    if (!undoStack.current.length) return;
    redoStack.current = [{nodes:JSON.parse(JSON.stringify(nodes)), edges:JSON.parse(JSON.stringify(edges))}, ...redoStack.current.slice(0,49)];
    const p = undoStack.current.pop();
    setNodes(p.nodes); setEdges(p.edges);
    setCanUndo(undoStack.current.length>0); setCanRedo(true);
  }, [nodes, edges]);

  const redo = useCallback(() => {
    if (!redoStack.current.length) return;
    undoStack.current = [...undoStack.current.slice(-49), {nodes:JSON.parse(JSON.stringify(nodes)), edges:JSON.parse(JSON.stringify(edges))}];
    const n = redoStack.current.shift();
    setNodes(n.nodes); setEdges(n.edges);
    setCanUndo(true); setCanRedo(redoStack.current.length>0);
  }, [nodes, edges]);

  const saveWorkflow = useCallback(() => {
    const b = new Blob([JSON.stringify({nodes,edges,version:1},null,2)], {type:"application/json"});
    const u = URL.createObjectURL(b);
    const a = document.createElement("a"); a.href=u; a.download="workflow.json"; a.click(); URL.revokeObjectURL(u);
  }, [nodes, edges]);

  const loadWorkflow = useCallback(e => {
    const f = e.target.files?.[0]; if (!f) return;
    const r = new FileReader();
    r.onload = ev => { try { const {nodes:ns,edges:es}=JSON.parse(ev.target.result); snapshot(); setNodes(ns); setEdges(es); setSelId(null); } catch { alert("Invalid JSON"); } };
    r.readAsText(f); e.target.value="";
  }, [snapshot]);

  // Backend health check
  useEffect(() => {
    if (!backendMode) { setBackendHealth("idle"); return; }
    setBackendHealth("checking");
    const ctrl = new AbortController();
    fetch(`${backendUrl}/health`, {signal:ctrl.signal})
      .then(r => r.json())
      .then(d => setBackendHealth(d.status==="ok"?"ok":"error"))
      .catch(() => setBackendHealth("error"));
    return () => ctrl.abort();
  }, [backendMode, backendUrl]);

  const toCanvas = useCallback((cx, cy) => {
    const r = cRef.current?.getBoundingClientRect();
    return r ? {x:(cx-r.left-pan.x)/zoom, y:(cy-r.top-pan.y)/zoom} : {x:0,y:0};
  }, [pan, zoom]);

  const onWheel = useCallback(e => {
    e.preventDefault();
    const nz = Math.max(.25, Math.min(2.5, zoom*(e.deltaY<0?1.1:.9)));
    const r = cRef.current?.getBoundingClientRect(); if (!r) return;
    const mx=e.clientX-r.left, my=e.clientY-r.top;
    setPan(p => ({x:mx-(mx-p.x)*(nz/zoom), y:my-(my-p.y)*(nz/zoom)}));
    setZoom(nz);
  }, [zoom]);

  const onBgDown = useCallback(e => {
    if (e.button!==0) return;
    setCtxMenu(null);
    if (conn) { setConn(null); return; }
    if (e.shiftKey) {
      const cp = toCanvas(e.clientX, e.clientY);
      drag.current = {type:"selbox", sx:e.clientX, sy:e.clientY, cx:cp.x, cy:cp.y};
      return;
    }
    setSelId(null); setMultiSel(new Set()); setSelEdge(null);
    drag.current = {type:"pan", sx:e.clientX, sy:e.clientY, ox:pan.x, oy:pan.y};
  }, [pan, conn, toCanvas]);

  const onNodeDown = useCallback((e, id) => {
    if (multiSel.has(id) && multiSel.size>1) {
      const ip = {};
      nodes.forEach(n => { if (multiSel.has(n.id)) ip[n.id]={x:n.x,y:n.y}; });
      drag.current = {type:"multi", sx:e.clientX, sy:e.clientY, ip};
    } else {
      const n = nodes.find(n=>n.id===id);
      drag.current = {type:"node", id, sx:e.clientX, sy:e.clientY, ox:n.x, oy:n.y};
    }
  }, [nodes, multiSel]);

  const onMove = useCallback(e => {
    setMpos(toCanvas(e.clientX, e.clientY));
    if (!drag.current) return;
    const dx=(e.clientX-drag.current.sx)/zoom, dy=(e.clientY-drag.current.sy)/zoom;
    if (drag.current.type==="pan") setPan({x:drag.current.ox+dx*zoom, y:drag.current.oy+dy*zoom});
    else if (drag.current.type==="node") { const id=drag.current.id; setNodes(p=>p.map(n=>n.id===id?{...n,x:drag.current.ox+dx,y:drag.current.oy+dy}:n)); }
    else if (drag.current.type==="multi") setNodes(p=>p.map(n=>{ const ip=drag.current.ip[n.id]; return ip?{...n,x:ip.x+dx,y:ip.y+dy}:n; }));
    else if (drag.current.type==="selbox") { const cp=toCanvas(e.clientX,e.clientY); setSelBox({x1:Math.min(drag.current.cx,cp.x),y1:Math.min(drag.current.cy,cp.y),x2:Math.max(drag.current.cx,cp.x),y2:Math.max(drag.current.cy,cp.y)}); }
  }, [toCanvas, zoom]);

  const onUp = useCallback(() => {
    if (drag.current?.type==="node"||drag.current?.type==="multi") snapshot();
    if (drag.current?.type==="selbox" && selBox) {
      const sel = nodes.filter(n=>n.x<selBox.x2&&n.x+NW>selBox.x1&&n.y<selBox.y2&&n.y+NH>selBox.y1);
      if (sel.length) { setMultiSel(new Set(sel.map(n=>n.id))); setSelId(sel[0].id); }
      setSelBox(null);
    }
    drag.current = null;
  }, [snapshot, selBox, nodes]);

  const delNode = useCallback(id => {
    setNodes(p=>p.filter(n=>n.id!==id));
    setEdges(p=>p.filter(e=>e.fn!==id&&e.tn!==id));
    if (selId===id) setSelId(null);
    setMultiSel(p=>{ const s=new Set(p); s.delete(id); return s; });
  }, [selId]);

  const addNode = t => {
    snapshot();
    const x=-pan.x/zoom+400/zoom+Math.random()*60-30;
    const y=-pan.y/zoom+200/zoom+Math.random()*60-30;
    setNodes(p=>[...p, mkNode(t,x,y)]);
  };

  const startConn = useCallback((nid, pi) => setConn({nid,pi}), []);
  const endConn = useCallback((tnid, tpi) => {
    if (!conn||conn.nid===tnid) return setConn(null);
    if (!edges.find(e=>e.fn===conn.nid&&e.fp===conn.pi&&e.tn===tnid&&e.tp===tpi)) {
      snapshot();
      setEdges(p=>[...p, {id:`e_${Date.now()}`,fn:conn.nid,fp:conn.pi,tn:tnid,tp:tpi}]);
    }
    setConn(null);
  }, [conn, edges, snapshot]);

  useEffect(() => {
    const h = e => {
      const tag = document.activeElement?.tagName;
      const typing = tag==="INPUT"||tag==="TEXTAREA"||tag==="SELECT";
      if (e.key==="Escape") { setConn(null); setCtxMenu(null); setMultiSel(new Set()); setSelId(null); return; }
      if (typing) return;
      if (e.key==="Delete"||e.key==="Backspace") {
        if (multiSel.size>1) { snapshot(); [...multiSel].forEach(id=>delNode(id)); setMultiSel(new Set()); return; }
        if (selId) { snapshot(); delNode(selId); return; }
      }
      if (e.ctrlKey||e.metaKey) {
        if (e.key==="a") { e.preventDefault(); setMultiSel(new Set(nodes.map(n=>n.id))); return; }
        if (e.key==="z"&&!e.shiftKey) { e.preventDefault(); undo(); return; }
        if (e.key==="y"||(e.key==="z"&&e.shiftKey)) { e.preventDefault(); redo(); return; }
        if (e.key==="s") { e.preventDefault(); saveWorkflow(); return; }
        if (e.key==="d"&&selId) { e.preventDefault(); snapshot(); const src=nodes.find(n=>n.id===selId); if(!src)return; const nn=mkNode(src.type,src.x+40,src.y+40,{label:src.label+" copy",config:JSON.parse(JSON.stringify(src.config))}); setNodes(p=>[...p,nn]); setSelId(nn.id); return; }
      }
      if (e.key==="="||e.key==="+") setZoom(z=>Math.min(2.5,+(z*1.15).toFixed(2)));
      if (e.key==="-") setZoom(z=>Math.max(.25,+(z*.87).toFixed(2)));
      if (e.key==="0") { setZoom(1); setPan({x:40,y:20}); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [selId, multiSel, conn, nodes, undo, redo, saveWorkflow, snapshot, delNode]);

  const runEngine = useCallback(async (opts={}) => {
    const {startId, singleId, preResults={}} = opts;
    setRunning(true); setShowLog(true);
    const addLog = (msg,type="info") => setLog(p=>[...p,{msg,type,t:new Date().toLocaleTimeString()}]);
    setLog([]);
    const lv = {...vars};
    try {
      if (!singleId) setNodes(p=>p.map(n=>({...n,status:"idle",result:null,error:null})));
      const outE={}, inE2={};
      edges.forEach(e=>{ (outE[e.fn]=outE[e.fn]||[]).push(e); (inE2[e.tn]=inE2[e.tn]||[]).push(e); });
      const results = {...preResults};
      const queue = startId?[startId]:singleId?[singleId]:nodes.filter(n=>ND[n.type]?.ins===0).map(n=>n.id);
      const visited = new Set();
      while (queue.length>0) {
        const nid = queue.shift();
        if (visited.has(nid)) continue;
        const deps = inE2[nid]||[];
        if (!singleId&&deps.length&&!deps.every(e=>results[e.fn]!==undefined)) { queue.push(nid); continue; }
        visited.add(nid);
        const node = nodes.find(n=>n.id===nid); if (!node||node.type==="note") continue;
        addLog(`Running: ${node.label}`, "run");
        setNodes(p=>p.map(n=>n.id===nid?{...n,status:"running"}:n));
        try {
          const inp = deps.map(e=>results[e.fn]||"").join("\n---\n");
          let result;
          if (node.type==="loop") {
            let items = inp.includes("|||SPLIT|||") ? inp.split("|||SPLIT|||").map(s=>s.trim()).filter(Boolean) : [];
            if (!items.length) { try { const p2=JSON.parse(inp); items=Array.isArray(p2)?p2:[inp]; } catch { items=[inp]; } }
            items = items.slice(0, node.config.maxItems||20);
            const parts = [];
            for (let i=0; i<items.length; i++) {
              setNodes(p=>p.map(n=>n.id===nid?{...n,status:"running",result:`${i+1}/${items.length}`}:n));
              const item = items[i];
              if (node.config.mode==="code") { try { const fn=new Function("item","index","vars",node.config.code); parts.push(String(fn(item,i,lv)||"")); } catch(e2) { parts.push(`[Error:${e2.message}]`); } }
              else if (node.config.mode==="template") parts.push(subVars(node.config.template.replace(/\{\{item\}\}/g,item), lv));
              else { const k=apiKeys[node.config.provider]; parts.push(await callAI({...node.config,systemPrompt:subVars(node.config.systemPrompt||"",lv)}, k, subVars((node.config.prompt||"{{item}}").replace(/\{\{item\}\}/g,item), lv), backendUrl, backendMode)); }
            }
            result = node.config.outputFormat==="array" ? JSON.stringify(parts) : parts.join((node.config.separator||"\\n---\\n").replace(/\\n/g,"\n"));
          } else {
            result = await execNode(node, inp, apiKeys, lv, backendUrl, backendMode);
            if (node.type==="setvar") lv[node.config.varName]=inp;
          }
          results[nid]=result;
          setNodes(p=>p.map(n=>n.id===nid?{...n,status:"done",result}:n));
          setEdgeData(p=>{ const nx={...p}; (outE[nid]||[]).forEach(e=>{ nx[e.id]=String(result); }); return nx; });
          addLog(`✓ ${node.label}: ${String(result).substring(0,60)}${String(result).length>60?"…":""}`, "done");
          if (!singleId) (outE[nid]||[]).forEach(e=>{ if(!visited.has(e.tn)) queue.push(e.tn); });
        } catch(e2) {
          setNodes(p=>p.map(n=>n.id===nid?{...n,status:"error",error:e2.message}:n));
          addLog(`✗ ${node.label}: ${e2.message}`, "error");
        }
      }
      setVars({...lv});
      addLog(`Complete · ${Object.keys(lv).length} vars`, "done");
    } catch(e2) { setLog(p=>[...p,{msg:`Fatal: ${e2.message}`,type:"error",t:""}]); }
    setRunning(false);
  }, [nodes, edges, apiKeys, vars, backendUrl, backendMode]);

  const initStep = useCallback((startId) => {
    stepResultsRef.current={}; stepVarsRef.current={};
    setNodes(p=>p.map(n=>({...n,status:"idle",result:null,error:null})));
    setEdgeData({}); setStepDone(new Set());
    const ready = startId ? [startId] : nodes.filter(n=>ND[n.type]?.ins===0).map(n=>n.id);
    setStepQueue(ready); setStepMode(true); setShowLog(true);
    setLog([{msg:`Step mode${startId?` from: ${nodes.find(n=>n.id===startId)?.label}`:""}`,type:"run",t:new Date().toLocaleTimeString()}]);
  }, [nodes]);

  const execStep = useCallback(async () => {
    if (!stepQueue.length||stepRun) return;
    const nid = stepQueue[0];
    const node = nodes.find(n=>n.id===nid); if (!node) { setStepQueue(p=>p.slice(1)); return; }
    setStepRun(true); setNodes(p=>p.map(n=>n.id===nid?{...n,status:"running"}:n));
    const inE2={}; edges.forEach(e=>{ (inE2[e.tn]=inE2[e.tn]||[]).push(e); });
    const deps = inE2[nid]||[];
    const inp = deps.map(e=>stepResultsRef.current[e.fn]||"").join("\n---\n");
    try {
      const result = await execNode(node, inp, apiKeys, stepVarsRef.current, backendUrl, backendMode);
      stepResultsRef.current[nid]=result;
      setNodes(p=>p.map(n=>n.id===nid?{...n,status:"done",result}:n));
      setEdgeData(p=>{ const nx={...p}; edges.filter(e=>e.fn===nid).forEach(e=>{ nx[e.id]=String(result); }); return nx; });
      setLog(p=>[...p,{msg:`✓ ${node.label}: ${String(result).substring(0,60)}…`,type:"done",t:new Date().toLocaleTimeString()}]);
      const outE={}; edges.forEach(e=>{ (outE[e.fn]=outE[e.fn]||[]).push(e); });
      const newDone = new Set([...stepDone, nid]); setStepDone(newDone);
      const inE3={}; edges.forEach(e=>{ (inE3[e.tn]=inE3[e.tn]||[]).push(e); });
      const newReady = (outE[nid]||[]).map(e=>e.tn).filter(tid=>!newDone.has(tid)&&(inE3[tid]||[]).every(e=>newDone.has(e.fn))&&!stepQueue.slice(1).includes(tid));
      const nq = [...stepQueue.slice(1), ...newReady]; setStepQueue(nq);
      if (!nq.length) { setStepMode(false); setVars({...stepVarsRef.current}); setLog(p=>[...p,{msg:"Step complete",type:"done",t:new Date().toLocaleTimeString()}]); }
    } catch(e2) {
      setNodes(p=>p.map(n=>n.id===nid?{...n,status:"error",error:e2.message}:n));
      setLog(p=>[...p,{msg:`✗ ${node.label}: ${e2.message}`,type:"error",t:new Date().toLocaleTimeString()}]);
      setStepQueue(p=>p.slice(1));
    }
    setStepRun(false);
  }, [stepQueue, stepDone, nodes, edges, apiKeys, stepRun]);

  const selNode = nodes.find(n=>n.id===selId);
  const svgEdges = edges.map(e => {
    const fn=nodes.find(n=>n.id===e.fn), tn=nodes.find(n=>n.id===e.tn);
    if (!fn||!tn) return null;
    const a=outP(fn,e.fp), b=inP(tn,e.tp);
    const hasData = !!edgeData[e.id];
    return {...e, path:bez(a,b), color:ND[fn.type]?.color||N.f3, a, b, hasData, fromLabel:fn.label, toLabel:tn.label};
  }).filter(Boolean);
  let prevEdge = null;
  if (conn) { const fn=nodes.find(n=>n.id===conn.nid); if(fn) { const a=outP(fn,conn.pi); prevEdge=bez(a,mpos); } }

  return (
    <div style={{height:"100vh",display:"flex",flexDirection:"column",background:N.n0,color:N.n4,fontFamily:"JetBrains Mono,monospace",overflow:"hidden"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:5px;height:5px} ::-webkit-scrollbar-track{background:${N.n1}} ::-webkit-scrollbar-thumb{background:${N.n3};border-radius:4px}
        @keyframes sweep{0%{background-position:-200% 0}100%{background-position:200% 0}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
        @keyframes glow{0%,100%{opacity:.7}50%{opacity:1}}
        select option{background:${N.n1};color:${N.n5}}
        input:focus,textarea:focus,select:focus{outline:none;border-color:${N.f3}!important}
        button{font-family:inherit} button:hover{opacity:.85} ::placeholder{color:#6B788A!important}
      `}</style>

      {/* ── Toolbar ── */}
      <div style={{height:54,background:N.n1,borderBottom:`1px solid ${N.n2}`,display:"flex",alignItems:"center",padding:"0 16px",gap:8,flexShrink:0}}>
        <div style={{display:"flex",alignItems:"center",gap:8,marginRight:4}}>
          <div style={{width:28,height:28,borderRadius:7,background:`linear-gradient(135deg,${N.f2},${N.f4})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:14,color:N.n0,fontWeight:700}}>H</div>
          <div>
            <div style={{fontSize:12,fontWeight:700,color:N.n5,letterSpacing:1}}>Hermes Workflows</div>
            <div style={{fontSize:8,color:N.n3,letterSpacing:2}}>VISUAL AUTOMATION</div>
          </div>
        </div>
        <div style={{width:1,height:28,background:N.n2}}/>
        {[
          {l:"▶ Run",   ok:!running&&!stepMode, c:N.a4, a:()=>runEngine()},
          {l:"◈ Step",  ok:!running&&!stepMode, c:N.f2, a:()=>initStep()},
          {l:"■ Stop",  ok:running,             c:N.a1, a:()=>setRunning(false)},
          {l:"↺ Reset", ok:!running&&!stepMode, c:N.a3, a:()=>{setNodes(p=>p.map(n=>({...n,status:"idle",result:null,error:null})));setEdgeData({});}},
          {l:"✕ Clear", ok:!running&&!stepMode, c:N.a2, a:()=>{snapshot();setNodes(DEFAULT_NODES);setEdges(DEFAULT_EDGES);setLog([]);setSelId(dn1.id);setEdgeData({});}},
        ].map(b => (
          <button key={b.l} onClick={b.ok?b.a:undefined} style={BS(b.ok,b.c)}>{b.l}</button>
        ))}
        {running && <span style={{fontSize:9,color:N.a3,animation:"glow 1s infinite",letterSpacing:1}}>● RUNNING</span>}
        <div style={{width:1,height:28,background:N.n2}}/>
        <button onClick={undo} disabled={!canUndo} style={BS(canUndo,N.f3)}>⟵ Undo</button>
        <button onClick={redo} disabled={!canRedo} style={BS(canRedo,N.f3)}>Redo ⟶</button>
        <div style={{width:1,height:28,background:N.n2}}/>
        <button onClick={saveWorkflow} title="Ctrl+S" style={BS(true,N.a4)}>↓ Save</button>
        <button onClick={()=>loadRef.current?.click()} style={BS(true,N.f1)}>↑ Load</button>
        <input ref={loadRef} type="file" accept=".json" onChange={loadWorkflow} style={{display:"none"}}/>
        <button onClick={()=>{setExpCode(genPy(nodes,edges));setShowExp(true);}} style={BS(true,N.a5)}>↗ Py</button>
        <div style={{flex:1}}/>
        <div style={{display:"flex",alignItems:"center",gap:4,background:N.n0,border:`1px solid ${N.n2}`,borderRadius:6,padding:"2px 4px"}}>
          <button onClick={()=>setZoom(z=>Math.max(.25,+(z*.87).toFixed(2)))} style={{background:"none",border:"none",color:N.n4,fontSize:14,cursor:"pointer",padding:"2px 6px",lineHeight:1}}>−</button>
          <span onClick={()=>{setZoom(1);setPan({x:40,y:20});}} style={{fontSize:10,color:N.n4,minWidth:38,textAlign:"center",cursor:"pointer"}}>{Math.round(zoom*100)}%</span>
          <button onClick={()=>setZoom(z=>Math.min(2.5,+(z*1.15).toFixed(2)))} style={{background:"none",border:"none",color:N.n4,fontSize:14,cursor:"pointer",padding:"2px 6px",lineHeight:1}}>+</button>
        </div>
        <button onClick={()=>setShowLog(v=>!v)} style={BS(showLog,N.f3)}>LOG{log.length>0?` (${log.length})`:""}</button>
        <button onClick={()=>setShowVars(v=>!v)} style={BS(showVars,N.f1)}>VARS{Object.keys(vars).length>0?` (${Object.keys(vars).length})`:""}</button>
        <button onClick={()=>setShowKeys(v=>!v)} style={{...BS(showKeys,N.a4),display:"flex",alignItems:"center",gap:5}}>
          <span>KEYS</span>
          {backendMode&&<span style={{width:6,height:6,borderRadius:"50%",background:backendHealth==="ok"?N.a4:backendHealth==="error"?N.a1:N.a3,display:"inline-block"}}/>}
        </button>
        <button onClick={()=>setShowMap(v=>!v)} title="Toggle minimap" style={BS(showMap,N.f1)}>⊡</button>
      </div>

      {/* ── Keys + Backend panel ── */}
      {showKeys && (
        <div style={{position:"absolute",top:55,right:16,zIndex:300,background:N.n1,border:`1px solid ${N.n2}`,borderRadius:12,
          width:380,maxHeight:"88vh",display:"flex",flexDirection:"column",
          boxShadow:"0 16px 48px rgba(0,0,0,.6)",animation:"fadeUp .15s ease"}}>

          {/* Header */}
          <div style={{padding:"14px 18px 10px",borderBottom:`1px solid ${N.n2}`,flexShrink:0}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              <div style={{fontSize:12,color:N.n5,fontWeight:700}}>API Keys & Backend</div>
              <button onClick={()=>setShowKeys(false)} style={{background:"none",border:"none",color:N.n3,cursor:"pointer",fontSize:16,lineHeight:1}}>×</button>
            </div>
            <div style={{fontSize:9,color:N.n3,marginTop:3}}>{Object.values(apiKeys).filter(v=>v&&v!=="ollama").length} of {Object.keys(PROVS).filter(k=>!PROVS[k].noKey).length} keys set</div>
          </div>

          <div style={{overflowY:"auto",flex:1}}>
            {/* ── Backend section ── */}
            <div style={{padding:"14px 18px",borderBottom:`1px solid ${N.n2}`,background:backendMode?`${N.f2}08`:N.n0}}>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10}}>
                <div style={{width:8,height:8,borderRadius:"50%",background:backendHealth==="ok"?N.a4:backendHealth==="error"?N.a1:backendHealth==="checking"?N.a3:N.n3,
                  boxShadow:backendHealth==="ok"?`0 0 6px ${N.a4}`:backendHealth==="error"?`0 0 6px ${N.a1}`:"none",transition:"all .3s"}}/>
                <div style={{fontSize:11,color:N.n5,fontWeight:600}}>FastAPI Backend</div>
                <div style={{marginLeft:"auto",display:"flex",alignItems:"center",gap:6}}>
                  <span style={{fontSize:9,color:backendMode?N.f2:N.n3}}>{backendMode?"ON":"OFF"}</span>
                  <div onClick={()=>setBackendMode(v=>!v)} style={{width:36,height:20,borderRadius:10,background:backendMode?N.f2:N.n3,cursor:"pointer",position:"relative",transition:"background .2s"}}>
                    <div style={{position:"absolute",top:2,left:backendMode?18:2,width:16,height:16,borderRadius:"50%",background:"white",transition:"left .2s"}}/>
                  </div>
                </div>
              </div>
              <div style={{fontSize:9,color:backendHealth==="ok"?N.a4:backendHealth==="error"?N.a1:N.n3,marginBottom:8,letterSpacing:.5}}>
                {backendHealth==="ok"?"✓ Connected to Hermes backend":backendHealth==="error"?"✗ Cannot reach backend — is server.py running?":backendHealth==="checking"?"Checking...":"Routes AI calls through your local Python server"}
              </div>
              <Fld label="BACKEND URL">
                <input value={backendUrl} onChange={e=>setBackendUrl(e.target.value)} style={{...IS,background:N.n1}} placeholder="http://localhost:8000"/>
              </Fld>
              <div style={{display:"flex",gap:8}}>
                <button onClick={()=>{setBackendHealth("checking");fetch(`${backendUrl}/health`).then(r=>r.json()).then(d=>setBackendHealth(d.status==="ok"?"ok":"error")).catch(()=>setBackendHealth("error"));}}
                  style={{...BS(true,N.f2),flex:1,textAlign:"center"}}>⟳ Test Connection</button>
                <button onClick={()=>{const code=genServer();const b=new Blob([code],{type:"text/plain"});const u=URL.createObjectURL(b);const a=document.createElement("a");a.href=u;a.download="server.py";a.click();URL.revokeObjectURL(u);}}
                  style={{...BS(true,N.a4),flex:1,textAlign:"center"}}>↓ Download server.py</button>
              </div>
              <div style={{marginTop:8,fontSize:8,color:N.n3,lineHeight:1.6}}>
                <code style={{background:N.n0,padding:"3px 6px",borderRadius:3,fontSize:8,color:N.f1}}>pip install fastapi uvicorn httpx python-dotenv</code>
                <span style={{marginLeft:6}}>→ python server.py</span>
              </div>
            </div>

            {/* ── Provider keys by category ── */}
            <div style={{padding:"10px 18px 18px"}}>
              <div style={{fontSize:9,color:N.n3,letterSpacing:2,marginBottom:12}}>API KEYS (stored in memory only)</div>
              {Object.entries(PROV_CATS).map(([cat,cv]) => {
                const provs = Object.entries(PROVS).filter(([,v])=>v.cat===cat);
                return (
                  <div key={cat} style={{marginBottom:16}}>
                    <div style={{fontSize:9,color:cv.color,letterSpacing:2,fontWeight:700,marginBottom:8,display:"flex",alignItems:"center",gap:6}}>
                      <div style={{flex:1,height:1,background:`${cv.color}30`}}/>
                      {cv.label.toUpperCase()}
                      <div style={{flex:1,height:1,background:`${cv.color}30`}}/>
                    </div>
                    {provs.map(([k,v]) => (
                      <div key={k} style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                        <div style={{width:6,height:6,borderRadius:"50%",background:apiKeys[k]&&apiKeys[k]!=="ollama"?N.a4:N.n3,flexShrink:0}}/>
                        <div style={{width:90,fontSize:9,color:N.n5,flexShrink:0}}>
                          <div style={{fontWeight:500}}>{v.name}</div>
                          {apiKeys[k]&&apiKeys[k]!=="ollama"&&<div style={{fontSize:8,color:N.a4}}>✓ set</div>}
                        </div>
                        <input
                          type="password"
                          value={apiKeys[k]||""}
                          placeholder={v.noKey?"local · no key needed":"sk-…"}
                          disabled={v.noKey}
                          onChange={e=>setApiKeys(p=>({...p,[k]:e.target.value}))}
                          style={{...IS,background:N.n0,flex:1,fontSize:10,padding:"5px 8px",opacity:v.noKey?.6:1}}/>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Vars panel */}
      {showVars && (
        <div style={{position:"absolute",top:55,right:showKeys?396:16,zIndex:300,background:N.n1,border:`1px solid ${N.n2}`,borderRadius:10,padding:16,width:280,boxShadow:"0 16px 48px rgba(0,0,0,.5)",animation:"fadeUp .15s ease",maxHeight:"70vh",overflowY:"auto"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12}}>
            <div style={{fontSize:11,color:N.n5,fontWeight:600}}>Variables</div>
            <button onClick={()=>setVars({})} style={{background:"none",border:`1px solid ${N.a1}40`,borderRadius:4,padding:"2px 8px",color:N.a1,fontSize:8,cursor:"pointer"}}>CLEAR ALL</button>
          </div>
          {Object.keys(vars).length===0 ? (
            <div style={{fontSize:10,color:N.n3,textAlign:"center",padding:"12px 0"}}>No variables set yet</div>
          ) : Object.entries(vars).map(([k,val]) => (
            <div key={k} style={{marginBottom:10,background:N.n0,borderRadius:6,padding:"8px 10px",border:`1px solid ${N.n2}`}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:3}}>
                <span style={{fontSize:10,color:N.f3,fontWeight:600,fontFamily:"monospace"}}>{"{{"+k+"}}"}</span>
                <button onClick={()=>setVars(p=>{const n={...p};delete n[k];return n;})} style={{background:"none",border:"none",color:N.n3,cursor:"pointer",fontSize:11}}>×</button>
              </div>
              <div style={{fontSize:10,color:N.n4,fontFamily:"monospace",whiteSpace:"pre-wrap",maxHeight:60,overflow:"hidden"}}>{String(val).substring(0,120)}</div>
            </div>
          ))}
        </div>
      )}

      <div style={{flex:1,display:"flex",overflow:"hidden",minHeight:0}}>
        {/* ── Palette ── */}
        <div style={{width:172,flexShrink:0,background:N.n1,borderRight:`1px solid ${N.n2}`,overflowY:"auto",padding:"12px 10px"}}>
          <div style={{fontSize:8,color:N.n3,letterSpacing:3,marginBottom:12,paddingLeft:2}}>ADD NODES</div>
          {Object.entries(CATS).map(([cat,cv]) => (
            <div key={cat} style={{marginBottom:12}}>
              <div style={{fontSize:8,letterSpacing:2,marginBottom:5,paddingLeft:4,color:cv.color,fontWeight:700}}>{cv.label.toUpperCase()}</div>
              {Object.entries(ND).filter(([,d])=>d.cat===cat).map(([type,def]) => (
                <div key={type} onClick={()=>addNode(type)}
                  style={{display:"flex",alignItems:"center",gap:8,padding:"7px 10px",marginBottom:3,background:N.n0,border:`1px solid ${N.n2}`,borderRadius:7,cursor:"pointer",userSelect:"none",transition:"all .15s"}}
                  onMouseEnter={e=>{e.currentTarget.style.background=N.n2;e.currentTarget.style.borderColor=def.color+"60";}}
                  onMouseLeave={e=>{e.currentTarget.style.background=N.n0;e.currentTarget.style.borderColor=N.n2;}}>
                  <span style={{fontSize:13,color:def.color,width:18,textAlign:"center"}}>{def.icon}</span>
                  <span style={{fontSize:10.5,color:N.n5,fontWeight:500}}>{def.label}</span>
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* ── Canvas ── */}
        <div ref={cRef}
          style={{flex:1,position:"relative",overflow:"hidden",cursor:"default",background:N.n0,
            backgroundImage:`radial-gradient(circle, ${N.n2} 1px, transparent 1px)`,
            backgroundSize:`${26*zoom}px ${26*zoom}px`,
            backgroundPosition:`${pan.x%(26*zoom)}px ${pan.y%(26*zoom)}px`}}
          onMouseDown={onBgDown} onMouseMove={onMove} onMouseUp={onUp} onWheel={onWheel}>

          <div style={{position:"absolute",top:0,left:0,transform:`translate(${pan.x}px,${pan.y}px) scale(${zoom})`,transformOrigin:"0 0"}}>
            <svg style={{position:"absolute",top:-3000,left:-3000,width:9000,height:9000,pointerEvents:"none",overflow:"visible"}}>
              <g transform="translate(3000,3000)">
                {svgEdges.map(e => {
                  const isDone = stepMode && stepDone.has(e.fn);
                  const ec = stepMode ? (isDone?N.a4:N.n3) : e.color;
                  return (
                    <g key={e.id}>
                      {e.hasData && <path d={e.path} stroke={`${e.color}25`} strokeWidth={10} fill="none" />}
                      <path d={e.path} stroke={`${ec}35`} strokeWidth={6} fill="none" />
                      <path d={e.path} stroke={ec} strokeWidth={e.hasData?2.5:1.5} fill="none" opacity={stepMode&&!isDone?.3:.85} strokeLinecap="round" strokeDasharray={e.hasData?"none":"4,3"} />
                      <circle cx={e.b.x} cy={e.b.y} r={e.hasData?5:3.5} fill={ec} opacity={stepMode&&!isDone?.3:.9} />
                      {selEdge===e.id && <path d={e.path} stroke={N.f2} strokeWidth={4} fill="none" opacity={.5} strokeLinecap="round" />}
                      <path d={e.path} stroke="transparent" strokeWidth={16} fill="none"
                        style={{pointerEvents:"stroke",cursor:"pointer"}}
                        onClick={ev => {
                          if (ev.shiftKey) { snapshot(); setEdges(p=>p.filter(x=>x.id!==e.id)); setSelEdge(null); }
                          else { setSelEdge(selEdge===e.id?null:e.id); setSelId(null); setMultiSel(new Set()); }
                        }}
                        onMouseEnter={ev => { if(!e.hasData)return; setHoverEdge({id:e.id,x:ev.clientX,y:ev.clientY,data:edgeData[e.id]||"",fromLabel:e.fromLabel,toLabel:e.toLabel}); }}
                        onMouseMove={ev => { if(hoverEdge?.id===e.id) setHoverEdge(p=>({...p,x:ev.clientX,y:ev.clientY})); }}
                        onMouseLeave={() => setHoverEdge(null)} />
                    </g>
                  );
                })}
                {prevEdge && <path d={prevEdge} stroke={`${N.f2}70`} strokeWidth={2} fill="none" strokeDasharray="6,4" strokeLinecap="round" />}
                {selBox && <rect x={selBox.x1} y={selBox.y1} width={selBox.x2-selBox.x1} height={selBox.y2-selBox.y1} fill={`${N.f3}10`} stroke={N.f3} strokeWidth={1.5} strokeDasharray="6,3" rx={4} />}
              </g>
            </svg>

            {nodes.map(node => (
              <div key={node.id}>
                {node.type==="note" ? (
                  <div onMouseDown={e=>{e.stopPropagation();if(e.button===0){onNodeDown(e,node.id);setSelId(node.id);}}}
                    style={{position:"absolute",left:node.x,top:node.y,width:200,minHeight:80,background:`${node.config.color||N.a3}18`,border:`1.5px solid ${selId===node.id?node.config.color||N.a3:`${node.config.color||N.a3}50`}`,borderRadius:8,padding:"10px 12px",cursor:"grab",userSelect:"none"}}>
                    <div style={{fontSize:10,color:node.config.color||N.a3,fontWeight:600,marginBottom:4}}>✎ NOTE</div>
                    <div style={{fontSize:11,color:N.n4,lineHeight:1.65,whiteSpace:"pre-wrap"}}>{node.config.text}</div>
                  </div>
                ) : (
                  <NodeCard node={node}
                    selected={selId===node.id}
                    multiSel={multiSel.has(node.id)&&multiSel.size>1}
                    connecting={!!conn}
                    stepStatus={stepMode?(stepQueue[0]===node.id?"next":stepDone.has(node.id)?"done":stepQueue.includes(node.id)?"queued":"waiting"):null}
                    onSelect={id=>{setSelId(id);setSelEdge(null);if(!multiSel.has(id))setMultiSel(new Set());}}
                    onDragStart={onNodeDown}
                    onStartConn={startConn}
                    onEndConn={endConn}
                    onCtxMenu={(e,id)=>{setCtxMenu({nodeId:id,x:e.clientX,y:e.clientY});setSelId(id);}} />
                )}
                {selId===node.id && (
                  <div onMouseDown={e=>e.stopPropagation()}
                    onClick={e=>{e.stopPropagation();snapshot();delNode(node.id);}}
                    style={{position:"absolute",left:node.x+(node.type==="note"?192:NW)-8,top:node.y-8,zIndex:50,width:22,height:22,borderRadius:"50%",background:N.n1,border:`1.5px solid ${N.a1}`,display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",fontSize:13,color:N.a1,fontWeight:600}}>×</div>
                )}
              </div>
            ))}
          </div>

          {nodes.length===0 && (
            <div style={{position:"absolute",top:"50%",left:"50%",transform:"translate(-50%,-50%)",textAlign:"center",pointerEvents:"none",color:N.n3}}>
              <div style={{fontSize:40,marginBottom:12,opacity:.3}}>◈</div>
              <div style={{fontSize:11}}>Add nodes from the palette</div>
            </div>
          )}

          {multiSel.size>1 && (
            <div style={{position:"absolute",top:12,right:14,zIndex:60,background:`${N.a3}20`,border:`1px solid ${N.a3}50`,borderRadius:6,padding:"5px 12px",fontSize:9,color:N.a3,display:"flex",gap:10,alignItems:"center"}}>
              <span>{multiSel.size} selected</span>
              <button onClick={()=>{snapshot();[...multiSel].forEach(id=>delNode(id));setMultiSel(new Set());}} style={{background:`${N.a1}20`,border:`1px solid ${N.a1}40`,borderRadius:4,padding:"2px 8px",color:N.a1,fontSize:9,cursor:"pointer"}}>Delete all</button>
              <button onClick={()=>setMultiSel(new Set())} style={{background:"none",border:"none",color:N.n3,cursor:"pointer",fontSize:12}}>×</button>
            </div>
          )}

          {showMap && <Minimap nodes={nodes} edges={edges} pan={pan} zoom={zoom} containerRef={cRef} onPanTo={(x,y)=>setPan({x,y})} />}

          <div style={{position:"absolute",bottom:12,left:"50%",transform:"translateX(-50%)",fontSize:9,color:N.n4,letterSpacing:1,pointerEvents:"none",userSelect:"none",background:`${N.n1}ee`,border:`1px solid ${N.n2}`,borderRadius:20,padding:"5px 16px",whiteSpace:"nowrap"}}>
            Scroll to zoom · Shift+drag to select · Right-click for menu · Ctrl+A select all · 0 reset
          </div>

          {stepMode && (
            <div style={{position:"absolute",top:12,left:"50%",transform:"translateX(-50%)",background:N.n1,border:`1.5px solid ${N.a3}60`,borderRadius:10,padding:"10px 16px",display:"flex",alignItems:"center",gap:10,boxShadow:`0 4px 20px rgba(0,0,0,.5)`,zIndex:100,animation:"fadeUp .2s ease"}}>
              <div>
                <div style={{fontSize:9,color:N.a3,fontWeight:700,letterSpacing:2}}>STEP MODE</div>
                <div style={{fontSize:9,color:N.f3}}>{stepQueue.length>0?`next: ${nodes.find(n=>n.id===stepQueue[0])?.label||"…"}`:"all done"}</div>
              </div>
              <div style={{width:1,height:28,background:N.n2}}/>
              <button onClick={execStep} disabled={!stepQueue.length||stepRun} style={BS(stepQueue.length>0&&!stepRun,N.a3)}>{stepRun?"⏳ Running…":"▶ Execute Step"}</button>
              <button onClick={()=>setStepMode(false)} style={BS(true,N.a1)}>✕ Exit</button>
            </div>
          )}
        </div>

        {/* ── Right panel ── */}
        <div style={{width:286,flexShrink:0,background:N.n1,borderLeft:`1px solid ${N.n2}`,display:"flex",flexDirection:"column",overflow:"hidden"}}>
          <div style={{padding:"12px 18px 10px",borderBottom:`1px solid ${N.n2}`,flexShrink:0,display:"flex",alignItems:"center",gap:8}}>
            <div style={{fontSize:9,color:N.n3,letterSpacing:3}}>{selEdge&&!selNode?"EDGE INSPECTOR":"NODE PROPERTIES"}</div>
            {selEdge&&!selNode && <span style={{fontSize:8,color:N.f3,background:`${N.f3}15`,border:`1px solid ${N.f3}30`,padding:"1px 6px",borderRadius:8,marginLeft:"auto"}}>shift+click to delete</span>}
          </div>
          {selEdge&&!selNode ? (
            <EdgeInspector
              edge={edges.find(e=>e.id===selEdge)}
              nodes={nodes}
              edgeData={edgeData}
              onDelete={()=>{snapshot();setEdges(p=>p.filter(e=>e.id!==selEdge));setSelEdge(null);}}
              onClose={()=>setSelEdge(null)} />
          ) : (
            <ConfigPanel node={selNode} apiKeys={apiKeys} onChange={upd=>setNodes(p=>p.map(n=>n.id===upd.id?upd:n))} />
          )}
        </div>
      </div>

      {/* ── Log panel ── */}
      {showLog && (
        <div style={{height:155,background:N.n1,borderTop:`1px solid ${N.n2}`,display:"flex",flexDirection:"column",flexShrink:0}}>
          <div style={{height:34,display:"flex",alignItems:"center",padding:"0 16px",gap:10,borderBottom:`1px solid ${N.n2}`,flexShrink:0}}>
            <span style={{fontSize:9,color:N.n3,letterSpacing:3}}>EXECUTION LOG</span>
            <div style={{flex:1}}/>
            <button onClick={()=>setLog([])} style={{background:"none",border:"none",color:N.n3,fontSize:9,cursor:"pointer"}}>CLEAR</button>
            <button onClick={()=>setShowLog(false)} style={{background:"none",border:"none",color:N.n3,fontSize:16,cursor:"pointer",lineHeight:1}}>×</button>
          </div>
          <div style={{flex:1,overflowY:"auto",padding:"8px 16px"}}>
            {log.length===0 && <div style={{fontSize:10,color:N.n3,padding:"6px 0"}}>Press ▶ Run to execute.</div>}
            {log.map((l,i) => (
              <div key={i} style={{display:"flex",gap:12,fontSize:10,lineHeight:1.9,fontFamily:"JetBrains Mono,monospace",color:l.type==="error"?N.a1:l.type==="done"?N.a4:l.type==="run"?N.a3:N.n3}}>
                <span style={{color:N.f3,flexShrink:0}}>{l.t}</span>
                <span>{l.msg}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Context menu ── */}
      {ctxMenu && (
        <div style={{position:"fixed",left:ctxMenu.x,top:ctxMenu.y,zIndex:500,background:N.n1,border:`1px solid ${N.n2}`,borderRadius:8,padding:"4px",minWidth:200,boxShadow:"0 8px 28px rgba(0,0,0,.55)",animation:"fadeUp .12s ease"}}
          onMouseLeave={()=>setCtxMenu(null)}>
          {[
            {icon:"▶",label:"Run from here",color:N.a4,act:()=>{ const pr={}; nodes.forEach(n=>{if(n.result)pr[n.id]=n.result;}); runEngine({startId:ctxMenu.nodeId,preResults:pr}); setCtxMenu(null); }},
            {icon:"◈",label:"Step from here",color:N.f2,act:()=>{ initStep(ctxMenu.nodeId); setCtxMenu(null); }},
            {icon:"⚡",label:"Run this node only",color:N.a3,act:()=>{ runEngine({singleId:ctxMenu.nodeId}); setCtxMenu(null); }},
            null,
            {icon:"⎘",label:"Duplicate",color:N.n4,act:()=>{ snapshot(); const src=nodes.find(n=>n.id===ctxMenu.nodeId); if(!src)return; const nn=mkNode(src.type,src.x+40,src.y+40,{label:src.label+" copy",config:JSON.parse(JSON.stringify(src.config))}); setNodes(p=>[...p,nn]); setSelId(nn.id); setCtxMenu(null); }},
            {icon:"✕",label:"Delete",color:N.a1,act:()=>{ snapshot(); delNode(ctxMenu.nodeId); setCtxMenu(null); }},
          ].map((item,i) => item===null ? (
            <div key={i} style={{height:1,background:N.n2,margin:"3px 0"}} />
          ) : (
            <div key={i} onClick={item.act}
              style={{display:"flex",alignItems:"center",gap:10,padding:"8px 14px",fontSize:11,color:item.color||N.n4,cursor:"pointer",borderRadius:5,transition:"background .1s"}}
              onMouseEnter={e=>e.currentTarget.style.background=N.n2}
              onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
              <span style={{fontSize:12,width:16,textAlign:"center"}}>{item.icon}</span>
              {item.label}
            </div>
          ))}
        </div>
      )}

      {/* ── Python export modal ── */}
      {showExp && (
        <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,.7)",zIndex:600,display:"flex",alignItems:"center",justifyContent:"center",padding:24}}
          onClick={e=>{if(e.target===e.currentTarget)setShowExp(false);}}>
          <div style={{background:N.n1,border:`1px solid ${N.n2}`,borderRadius:12,width:"min(820px,96vw)",height:"min(640px,90vh)",display:"flex",flexDirection:"column",boxShadow:"0 24px 64px rgba(0,0,0,.7)"}}>
            <div style={{display:"flex",alignItems:"center",padding:"14px 20px",borderBottom:`1px solid ${N.n2}`,flexShrink:0}}>
              <div style={{fontSize:12,fontWeight:700,color:N.n5,marginRight:"auto"}}>🐍 Export as Python · {nodes.length} nodes</div>
              <button onClick={()=>navigator.clipboard.writeText(expCode)} style={{...BS(true,N.f2),marginRight:8}}>⎘ Copy</button>
              <button onClick={()=>{ const b=new Blob([expCode],{type:"text/plain"}); const u=URL.createObjectURL(b); const a=document.createElement("a"); a.href=u; a.download="workflow.py"; a.click(); URL.revokeObjectURL(u); }} style={{...BS(true,N.a4),marginRight:8}}>↓ .py</button>
              <button onClick={()=>setShowExp(false)} style={{background:"none",border:"none",color:N.n3,fontSize:18,cursor:"pointer"}}>×</button>
            </div>
            <textarea readOnly value={expCode} style={{flex:1,background:N.n0,border:"none",color:N.f1,fontSize:11,fontFamily:"JetBrains Mono,monospace",lineHeight:1.7,padding:"16px 20px",resize:"none"}} />
          </div>
        </div>
      )}

      {/* ── Edge hover tooltip ── */}
      {hoverEdge && (
        <div style={{position:"fixed",left:Math.min(hoverEdge.x+16,window.innerWidth-360),top:Math.max(hoverEdge.y-12,8),zIndex:500,background:N.n1,border:`1px solid ${N.f3}60`,borderRadius:8,padding:"10px 14px",maxWidth:340,pointerEvents:"none",boxShadow:"0 8px 28px rgba(0,0,0,.55)"}}>
          <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:6}}>
            <span style={{fontSize:9,color:N.f3,fontWeight:600}}>{hoverEdge.fromLabel}</span>
            <span style={{fontSize:11,color:N.n3}}>→</span>
            <span style={{fontSize:9,color:N.f3,fontWeight:600}}>{hoverEdge.toLabel}</span>
            <span style={{marginLeft:"auto",fontSize:8,color:N.n3,background:N.n2,padding:"1px 6px",borderRadius:8}}>{hoverEdge.data.length.toLocaleString()} chars</span>
          </div>
          <div style={{background:N.n0,borderRadius:5,padding:"8px 10px",border:`1px solid ${N.n2}`,fontSize:10.5,color:N.f1,fontFamily:"JetBrains Mono,monospace",whiteSpace:"pre-wrap",lineHeight:1.65,maxHeight:180,overflow:"hidden"}}>{hoverEdge.data.substring(0,400)}{hoverEdge.data.length>400?"…":""}</div>
        </div>
      )}
    </div>
  );
}

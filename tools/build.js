'use strict';
/* 生成单文件 index.html: 把 chenxing-ai.js 原样内联进模板, 零外部依赖。
 * 用法: node tools/build.js  (可复现) */
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
const engine = fs.readFileSync(path.join(root, 'chenxing-ai.js'), 'utf8');
const tpl = fs.readFileSync(path.join(__dirname, 'template.html'), 'utf8');
const out = tpl.replace('/*ENGINE*/', () => engine);
fs.writeFileSync(path.join(root, 'index.html'), out);
console.log('index.html 生成完成, 大小', (out.length / 1024).toFixed(1), 'KB');

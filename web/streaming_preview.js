// Nodes 2.0 renders send_progress_text() output as a markdown widget named
// "$$node-text-preview" whose grid row sizes to content, so long LLM outputs
// keep stretching the node taller. This caps the preview height, keeps it
// scrolled to the newest text while streaming, and adds a settings toggle to
// hide the preview box entirely.

import { app } from '../../scripts/app.js'

const PREVIEW_SELECTOR = '.comfy-markdown-content[aria-label="$$node-text-preview"]'
const HIDE_CLASS = 'h3-hide-streaming-preview'
const SETTING_ID = 'H3.StreamingPreview.Show'
const MAX_HEIGHT_PX = 200

const style = document.createElement('style')
style.textContent = `
${PREVIEW_SELECTOR} { max-height: ${MAX_HEIGHT_PX}px; }
body.${HIDE_CLASS} [data-testid="node-widget"]:has(${PREVIEW_SELECTOR}) { display: none; }
`
document.head.appendChild(style)

// Follow the stream: stay pinned to the bottom as text arrives, unless the
// user scrolled up to read earlier output.
function attachFollowScroll(el) {
  if (el.dataset.h3FollowScroll) return
  el.dataset.h3FollowScroll = '1'
  let follow = true
  el.addEventListener('scroll', () => {
    follow = el.scrollTop + el.clientHeight >= el.scrollHeight - 8
  }, { passive: true })
  new MutationObserver(() => {
    if (follow) el.scrollTop = el.scrollHeight
  }).observe(el, { childList: true, subtree: true, characterData: true })
}

function setVisible(show) {
  document.body.classList.toggle(HIDE_CLASS, !show)
}

app.registerExtension({
  name: 'h3.streamingPreview',
  settings: [
    {
      id: SETTING_ID,
      name: '显示流式文本预览 (streaming text preview)',
      category: ['H3', 'Streaming Preview', '显示流式文本预览'],
      tooltip: 'H3 的 LLM 节点（灵感生成、提示词增强等）执行时在节点上实时滚动显示生成文本；关闭后该预览框完全隐藏。Show the live LLM output box on H3 nodes while they run.',
      type: 'boolean',
      defaultValue: true,
      onChange: (value) => setVisible(value),
    },
  ],
  setup() {
    setVisible(app.ui.settings.getSettingValue(SETTING_ID) !== false)
    setInterval(() => {
      document.querySelectorAll(PREVIEW_SELECTOR).forEach(attachFollowScroll)
    }, 1000)
  },
})

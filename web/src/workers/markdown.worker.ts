import Katex from 'katex'
import MarkdownIt from 'markdown-it'

interface MarkdownRequest {
  id: string
  content: string
}

interface MarkdownResult {
  id: string
  html: string
}

const markdown = new MarkdownIt({
  html: false,
  linkify: false,
  typographer: false,
})

self.onmessage = (event: MessageEvent<MarkdownRequest>) => {
  const { id, content } = event.data
  const math: string[] = []
  const text = content.replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+)\$/g, (_, display, inline) => {
    const expression = String(display ?? inline ?? '')
    const rendered = Katex.renderToString(expression, {
      displayMode: display !== undefined,
      throwOnError: false,
      trust: false,
    })
    const token = `ARIE_KATEX_${math.length}_TOKEN`
    math.push(rendered)
    return token
  })
  const html = markdown.render(text).replace(/ARIE_KATEX_(\d+)_TOKEN/g, (_, index: string) => {
    return math[Number(index)] ?? ''
  })
  self.postMessage({ id, html } satisfies MarkdownResult)
}

const questionInput = document.getElementById('question');
const askButton = document.getElementById('askButton');
const answerBox = document.getElementById('answer');
const loading = document.getElementById('loading');
const sourcesList = document.getElementById('sources');

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    answerBox.textContent = 'Please enter a question first.';
    return;
  }

  loading.classList.remove('hidden');
  answerBox.textContent = '';
  sourcesList.innerHTML = '';

  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || data.answer || 'Request failed');
    }

    answerBox.textContent = data.answer || 'No answer returned.';
    if (Array.isArray(data.warnings) && data.warnings.length) {
      answerBox.textContent += `\n\nWarnings:\n${data.warnings.join('\n')}`;
    }

    const items = Array.isArray(data.sources) ? data.sources : [];
    if (!items.length) {
      const li = document.createElement('li');
      li.textContent = 'No source metadata was returned.';
      sourcesList.appendChild(li);
      return;
    }

    for (const source of items) {
      const li = document.createElement('li');
      const doc = source.doc_name || 'Unknown document';
      const pageNums = source.page_nums || (source.page_num ? [source.page_num] : []);
      const sectionText = source.section_ids && source.section_ids.length ? ` • ${source.section_ids.join(', ')}` : '';
      const title = document.createElement('strong');
      title.textContent = doc;
      li.appendChild(title);
      const location = [
        pageNums.length ? `p. ${pageNums.join(', ')}` : '',
        sectionText,
      ].filter(Boolean).join(' ');
      if (location) {
        li.appendChild(document.createTextNode(` ${location}`));
      }
      sourcesList.appendChild(li);
    }
  } catch (error) {
    answerBox.textContent = error.message || 'Something went wrong while searching.';
  } finally {
    loading.classList.add('hidden');
  }
}

askButton.addEventListener('click', askQuestion);
questionInput.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    askQuestion();
  }
});

const STORAGE_KEY = "todos";

const CATEGORY_LABELS = {
  work: "업무",
  personal: "개인",
  study: "공부",
};

const EDIT_ICON_PATH =
  "M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z";
const DELETE_ICON_PATH =
  "M6 7h12l-1 13.5a1 1 0 0 1-1 .95H8a1 1 0 0 1-1-.95L6 7zm9-3V3a1 1 0 0 0-1-1H10a1 1 0 0 0-1 1v1H4v2h16V4h-5z";
const SAVE_ICON_PATH = "M9 16.17L4.83 12l-1.41 1.41L9 19 21 7l-1.41-1.41z";
const CANCEL_ICON_PATH =
  "M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z";

const PROGRESS_ELEMENT_IDS = {
  work: "progressWork",
  personal: "progressPersonal",
  study: "progressStudy",
};

const CATEGORY_KEYWORDS = {
  work: [
    "회의", "미팅", "보고서", "보고", "프로젝트", "이메일", "메일",
    "발표", "계약", "출근", "회사", "클라이언트", "고객", "기획",
    "마감", "결재", "출장", "업무",
  ],
  personal: [
    "병원", "약국", "쇼핑", "장보기", "가족", "친구", "운동", "청소",
    "빨래", "은행", "생일", "여행", "식사", "저녁", "약속", "전화",
  ],
  study: [
    "공부", "시험", "과제", "강의", "수업", "독서", "책", "자격증",
    "스터디", "논문", "복습", "예습", "숙제", "학원",
  ],
};

function suggestCategory(text) {
  const normalized = text.trim();
  if (!normalized) return null;

  let bestCategory = null;
  let bestScore = 0;

  Object.entries(CATEGORY_KEYWORDS).forEach(([category, keywords]) => {
    const score = keywords.filter((keyword) =>
      normalized.includes(keyword)
    ).length;
    if (score > bestScore) {
      bestScore = score;
      bestCategory = category;
    }
  });

  return bestCategory;
}

let todos = [];
let editingId = null;
let currentFilter = "all";

function generateId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function saveTodos(todosToSave) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(todosToSave));
  } catch (error) {
    console.error("할 일을 저장하는 중 오류가 발생했습니다:", error);
  }
}

function loadTodos() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];

    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error("저장된 할 일 데이터를 읽는 중 오류가 발생했습니다:", error);
    return [];
  }
}

function addTodo(text, category) {
  const now = new Date().toISOString();
  const todo = {
    id: generateId(),
    text,
    category,
    completed: false,
    createdAt: now,
    updatedAt: now,
  };

  todos.push(todo);
  saveTodos(todos);
  renderTodos();

  return todo;
}

function toggleComplete(id) {
  const todo = todos.find((t) => t.id === id);
  if (!todo) return;

  todo.completed = !todo.completed;
  todo.updatedAt = new Date().toISOString();
  saveTodos(todos);
  renderTodos();
}

function handleDelete(id) {
  const todo = todos.find((t) => t.id === id);
  if (!todo) return;

  const confirmed = confirm(`"${todo.text}" 항목을 삭제하시겠습니까?`);
  if (!confirmed) return;

  todos = todos.filter((t) => t.id !== id);
  if (editingId === id) editingId = null;
  saveTodos(todos);
  renderTodos();
}

function startEdit(id) {
  editingId = id;
  renderTodos();
}

function cancelEdit() {
  editingId = null;
  renderTodos();
}

function commitEdit(id, rawText, category) {
  const text = rawText.trim();
  if (!text) return;

  const todo = todos.find((t) => t.id === id);
  if (!todo) return;

  todo.text = text;
  todo.category = category;
  todo.updatedAt = new Date().toISOString();
  editingId = null;
  saveTodos(todos);
  renderTodos();
}

function createIconButton(className, label, iconPath) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `icon-btn ${className}`;
  button.setAttribute("aria-label", label);
  button.innerHTML = `
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path fill="currentColor" d="${iconPath}"/>
    </svg>
  `;
  return button;
}

function createTodoElement(todo) {
  const li = document.createElement("li");
  li.className = `todo-item${todo.completed ? " is-completed" : ""}`;
  li.dataset.category = todo.category;
  li.dataset.id = todo.id;

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.className = "todo-checkbox";
  checkbox.checked = todo.completed;
  checkbox.setAttribute("aria-label", "완료 표시");
  checkbox.addEventListener("change", () => toggleComplete(todo.id));

  const text = document.createElement("span");
  text.className = "todo-text";
  text.textContent = todo.text;

  const tag = document.createElement("span");
  tag.className = `category-tag category-tag--${todo.category}`;
  tag.textContent = CATEGORY_LABELS[todo.category] ?? todo.category;

  const editBtn = createIconButton("edit-btn", "수정", EDIT_ICON_PATH);
  editBtn.addEventListener("click", () => startEdit(todo.id));

  const deleteBtn = createIconButton("delete-btn", "삭제", DELETE_ICON_PATH);
  deleteBtn.addEventListener("click", () => handleDelete(todo.id));

  li.appendChild(checkbox);
  li.appendChild(text);
  li.appendChild(tag);
  li.appendChild(editBtn);
  li.appendChild(deleteBtn);

  return li;
}

function createTodoEditElement(todo) {
  const li = document.createElement("li");
  li.className = "todo-item is-editing";
  li.dataset.category = todo.category;
  li.dataset.id = todo.id;

  const input = document.createElement("input");
  input.type = "text";
  input.className = "todo-edit-input";
  input.value = todo.text;

  const select = document.createElement("select");
  select.className = "todo-edit-category";
  Object.entries(CATEGORY_LABELS).forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    if (value === todo.category) option.selected = true;
    select.appendChild(option);
  });

  const commit = () => commitEdit(todo.id, input.value, select.value);

  const saveBtn = createIconButton("save-btn", "저장", SAVE_ICON_PATH);
  saveBtn.addEventListener("click", commit);

  const cancelBtn = createIconButton("cancel-btn", "취소", CANCEL_ICON_PATH);
  cancelBtn.addEventListener("click", cancelEdit);

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      commit();
    } else if (event.key === "Escape") {
      event.preventDefault();
      cancelEdit();
    }
  });

  li.appendChild(input);
  li.appendChild(select);
  li.appendChild(saveBtn);
  li.appendChild(cancelBtn);

  return li;
}

function updateProgress() {
  const total = todos.length;
  const completed = todos.filter((t) => t.completed).length;
  const percent = total === 0 ? 0 : Math.round((completed / total) * 100);

  document.getElementById("progressFill").style.width = `${percent}%`;
  document.getElementById("progressText").textContent =
    total === 0 ? "할 일 없음" : `${completed}/${total} 완료 · ${percent}%`;

  Object.entries(PROGRESS_ELEMENT_IDS).forEach(([category, elementId]) => {
    const categoryTodos = todos.filter((t) => t.category === category);
    const categoryCompleted = categoryTodos.filter((t) => t.completed).length;
    document.getElementById(elementId).textContent =
      `${CATEGORY_LABELS[category]} ${categoryCompleted}/${categoryTodos.length}`;
  });
}

function getVisibleTodos() {
  const filtered =
    currentFilter === "all"
      ? todos
      : todos.filter((todo) => todo.category === currentFilter);

  return [...filtered].sort(
    (a, b) => Number(a.completed) - Number(b.completed)
  );
}

function renderTodos() {
  const listEl = document.getElementById("todoList");
  const emptyEl = document.getElementById("emptyState");

  listEl.innerHTML = "";
  updateProgress();

  const visibleTodos = getVisibleTodos();

  if (visibleTodos.length === 0) {
    emptyEl.textContent =
      currentFilter === "all"
        ? "표시할 할 일이 없습니다. 새로운 할 일을 추가해보세요!"
        : `${CATEGORY_LABELS[currentFilter]} 카테고리에 할 일이 없어요.`;
    emptyEl.hidden = false;
    return;
  }

  emptyEl.hidden = true;
  visibleTodos.forEach((todo) => {
    const el =
      todo.id === editingId
        ? createTodoEditElement(todo)
        : createTodoElement(todo);
    listEl.appendChild(el);
  });

  if (editingId) {
    const editInput = listEl.querySelector(
      `[data-id="${editingId}"] .todo-edit-input`
    );
    if (editInput) {
      editInput.focus();
      editInput.select();
    }
  }
}

function setupFilterTabs() {
  const buttons = document.querySelectorAll(".filter-btn");

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      currentFilter = button.dataset.filter;
      buttons.forEach((b) => b.classList.toggle("is-active", b === button));
      renderTodos();
    });
  });
}

function setupAddForm() {
  const form = document.getElementById("addForm");
  const input = document.getElementById("todoInput");
  const categoryRadios = form.querySelectorAll('input[name="category"]');
  const suggestionHint = document.getElementById("categorySuggestionHint");
  const suggestionLabel = document.getElementById("suggestedCategoryLabel");
  let categoryAutoLocked = false;

  const hideSuggestionHint = () => {
    suggestionHint.hidden = true;
  };

  categoryRadios.forEach((radio) => {
    radio.addEventListener("click", () => {
      categoryAutoLocked = true;
      hideSuggestionHint();
    });
  });

  input.addEventListener("input", () => {
    if (categoryAutoLocked) return;

    const suggested = suggestCategory(input.value);
    if (!suggested) {
      hideSuggestionHint();
      return;
    }

    const target = form.querySelector(
      `input[name="category"][value="${suggested}"]`
    );
    if (target) target.checked = true;

    suggestionLabel.textContent = CATEGORY_LABELS[suggested];
    suggestionHint.hidden = false;
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();

    const text = input.value.trim();
    if (!text) return;

    const checkedCategory = form.querySelector(
      'input[name="category"]:checked'
    );
    const category = checkedCategory ? checkedCategory.value : "personal";

    addTodo(text, category);
    input.value = "";
    input.focus();
    categoryAutoLocked = false;
    hideSuggestionHint();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  todos = loadTodos();
  renderTodos();
  setupAddForm();
  setupFilterTabs();
});

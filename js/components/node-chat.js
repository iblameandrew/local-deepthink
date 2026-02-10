/**
 * NodeChat — Vanilla HTML5/JS reimplementation
 * 
 * Provides:
 *   - NodeChat.init(containerId, options)   — mount the chat UI
 *   - FluxEye, FluxConfirm helpers
 * 
 * Depends on: marked.js (already in project)
 */
const NodeChat = (() => {
    'use strict';

    /* ============================================================
       Utility helpers
       ============================================================ */
    const h = (tag, attrs = {}, ...children) => {
        const el = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs)) {
            if (k === 'style' && typeof v === 'object') {
                Object.assign(el.style, v);
            } else if (k.startsWith('on') && typeof v === 'function') {
                el.addEventListener(k.slice(2).toLowerCase(), v);
            } else if (k === 'className') {
                el.className = v;
            } else if (k === 'innerHTML') {
                el.innerHTML = v;
            } else {
                el.setAttribute(k, v);
            }
        }
        for (const c of children) {
            if (typeof c === 'string') el.appendChild(document.createTextNode(c));
            else if (c) el.appendChild(c);
        }
        return el;
    };

    const scrollToBottom = (el) => {
        el.scrollTop = el.scrollHeight;
    };

    /* SVG icons (inline, no dependency) */
    const icons = {
        send: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`,
        trash: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14a2 2 0 01-2 2H9a2 2 0 01-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"/></svg>`,
        refresh: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/></svg>`,
        brain: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a7 7 0 00-7 7c0 3 2 5.5 4 7.5S12 22 12 22s3-3.5 5-5.5 4-4.5 4-7.5a7 7 0 00-7-7z"/></svg>`,
    };

    /* ============================================================
       FluxEye — pure DOM builder
       ============================================================ */
    function createFluxEye(size = '') {
        const sizeClass = size ? ` flux-eye--${size}` : '';
        const root = h('div', { className: `flux-eye${sizeClass}` });

        root.appendChild(h('div', { className: 'flux-eye__bezel' },
            h('div', { className: 'flux-eye__bezel-inner' })
        ));
        root.appendChild(h('div', { className: 'flux-eye__ring' }));

        const iris = h('div', { className: 'flux-eye__iris' },
            h('div', { className: 'flux-eye__pupil' },
                h('div', { className: 'flux-eye__dot' })
            ),
            h('div', { className: 'flux-eye__overlay' })
        );
        root.appendChild(iris);
        root.appendChild(h('div', { className: 'flux-eye__highlight' }));

        /* API */
        root._setState = (state) => {
            root.classList.remove('is-active', 'is-animate', 'is-thinking', 'is-pulsar');
            if (state) {
                root.classList.add('is-active');
                root.classList.add(`is-${state}`);
            }
        };

        return root;
    }

    /* ============================================================
       FluxConfirm — dynamic modal
       ============================================================ */
    function showConfirm({ title, message, onConfirm, onCancel, confirmText = 'Confirm', cancelText = 'Cancel', variant = 'info' }) {
        const overlay = h('div', { className: 'flux-confirm-overlay' });

        const modal = h('div', { className: `flux-confirm flux-confirm--${variant}` });
        modal.appendChild(h('div', { className: 'flux-confirm__glow-top' }));
        modal.appendChild(h('div', { className: 'flux-confirm__glow-bottom' }));

        const body = h('div', { className: 'flux-confirm__body' });
        body.appendChild(h('h3', { className: 'flux-confirm__title' }, title));
        body.appendChild(h('p', { className: 'flux-confirm__message' }, message));

        const actions = h('div', { className: 'flux-confirm__actions' });

        const cancelBtn = h('button', {
            className: 'flux-btn flux-btn--secondary',
            onClick: () => { overlay.remove(); if (onCancel) onCancel(); }
        }, cancelText);

        const confirmBtn = h('button', {
            className: `flux-btn flux-btn--primary${variant === 'danger' ? ' flux-confirm__danger-btn' : ''}`,
            onClick: () => { overlay.remove(); if (onConfirm) onConfirm(); }
        }, confirmText);

        actions.appendChild(cancelBtn);
        actions.appendChild(confirmBtn);
        body.appendChild(actions);
        modal.appendChild(body);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        // Close on background click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) { overlay.remove(); if (onCancel) onCancel(); }
        });

        return overlay;
    }

    /* ============================================================
       Thinking Indicator
       ============================================================ */
    function createThinkingIndicator() {
        const dots = h('div', { className: 'nc-thinking__dots' },
            h('span'), h('span'), h('span')
        );
        const eye = createFluxEye('xs');
        eye._setState('thinking');

        return h('div', { className: 'nc-thinking' }, eye, dots);
    }

    /* ============================================================
       NodeChat Core
       ============================================================ */
    function init(containerId, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) { console.error(`NodeChat: container #${containerId} not found`); return null; }

        const opts = {
            title: options.title || 'Brainstorm',
            subtitle: options.subtitle || 'AI-powered thinking',
            placeholder: options.placeholder || 'What would you like to brainstorm?',
            onSend: options.onSend || null,          // async (message, chatInstance) => {}
            onClear: options.onClear || null,        // () => {}
            ...options
        };

        /* ---- State ---- */
        const state = {
            messages: [],       // { role: 'user'|'assistant', content: string }
            isStreaming: false,
        };

        /* ---- Build DOM ---- */
        const root = h('div', { className: 'node-chat nc-animate-expand-living' });

        // -- Header --
        const eyeHeader = createFluxEye('xs');
        const headerLeft = h('div', { className: 'node-chat__header-left' },
            eyeHeader,
            h('div', {},
                h('h3', { className: 'node-chat__title' }, opts.title),
                h('p', { className: 'node-chat__subtitle' }, opts.subtitle)
            )
        );

        const resetBtn = h('button', {
            className: 'flux-btn--icon',
            title: 'Reset',
            innerHTML: icons.refresh,
            onClick: () => handleClear()
        });

        const headerActions = h('div', { className: 'node-chat__header-actions' },
            resetBtn
        );
        const header = h('div', { className: 'node-chat__header' }, headerLeft, headerActions);

        // -- Messages --
        const messagesArea = h('div', { className: 'node-chat__messages nc-custom-scrollbar' });
        const emptyState = h('div', { className: 'node-chat__empty' },
            h('p', {}, 'Ask a question or share an idea to get started…')
        );
        messagesArea.appendChild(emptyState);

        // -- Input --
        const textarea = h('textarea', {
            className: 'node-chat__textarea',
            placeholder: opts.placeholder,
            rows: '1'
        });
        const sendBtn = h('button', {
            className: 'node-chat__send-btn',
            innerHTML: icons.send,
            title: 'Send'
        });

        const inputRow = h('div', { className: 'node-chat__input-row' }, textarea, sendBtn);
        const inputArea = h('div', { className: 'node-chat__input-area' }, inputRow);

        root.appendChild(header);
        root.appendChild(messagesArea);
        root.appendChild(inputArea);

        container.innerHTML = '';
        container.appendChild(root);

        /* ---- Auto-resize textarea ---- */
        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
        });

        /* ---- Keyboard handler ---- */
        textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
        });

        sendBtn.addEventListener('click', () => handleSend());

        /* ---- Helpers ---- */
        function clearEmpty() {
            const empty = messagesArea.querySelector('.node-chat__empty');
            if (empty) empty.remove();
        }

        function addUserBubble(text) {
            clearEmpty();
            const bubble = h('div', { className: 'nc-msg nc-msg--user' }, text);
            messagesArea.appendChild(bubble);
            scrollToBottom(messagesArea);
        }

        function addAIBubble() {
            clearEmpty();
            const bubble = h('div', { className: 'nc-msg nc-msg--ai' });
            messagesArea.appendChild(bubble);
            scrollToBottom(messagesArea);
            return bubble;
        }

        function addThinking() {
            clearEmpty();
            const indicator = createThinkingIndicator();
            indicator.setAttribute('data-thinking', 'true');
            messagesArea.appendChild(indicator);
            scrollToBottom(messagesArea);
            return indicator;
        }

        function removeThinking() {
            const el = messagesArea.querySelector('[data-thinking]');
            if (el) el.remove();
        }

        function renderMarkdown(text) {
            if (typeof marked !== 'undefined' && marked.parse) {
                return marked.parse(text);
            }
            // Fallback
            return text
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        }

        /* ---- Send handler ---- */
        async function handleSend() {
            const text = textarea.value.trim();
            if (!text || state.isStreaming) return;

            state.messages.push({ role: 'user', content: text });
            addUserBubble(text);
            textarea.value = '';
            textarea.style.height = 'auto';

            state.isStreaming = true;
            setUILocked(true);
            eyeHeader._setState('thinking');

            const thinkingEl = addThinking();

            if (opts.onSend) {
                try {
                    await opts.onSend(text, chatAPI);
                } catch (err) {
                    removeThinking();
                    const errBubble = addAIBubble();
                    errBubble.innerHTML = `<span style="color:#ef4444">Error: ${err.message}</span>`;
                }
            } else {
                // Default: echo back after 1s (demo)
                setTimeout(() => {
                    removeThinking();
                    const bubble = addAIBubble();
                    bubble.innerHTML = renderMarkdown(`You said: **${text}**\n\nConnect an \`onSend\` handler to make this functional.`);
                    finishStreaming();
                }, 1000);
            }
        }

        /* ---- Streaming API for external callers ---- */
        function startAIResponse() {
            removeThinking();
            const bubble = addAIBubble();
            eyeHeader._setState('thinking');
            return bubble;
        }

        function updateAIResponse(bubble, fullText) {
            bubble.innerHTML = renderMarkdown(fullText);
            scrollToBottom(messagesArea);
        }

        function finishStreaming(content) {
            state.isStreaming = false;
            setUILocked(false);
            eyeHeader._setState(null);
            if (content) {
                state.messages.push({ role: 'assistant', content });
            }
        }

        function appendAIMessage(text) {
            // For SSE-style: add a complete message at once
            clearEmpty();
            removeThinking();
            const bubble = addAIBubble();
            bubble.innerHTML = renderMarkdown(text);
            state.messages.push({ role: 'assistant', content: text });
            state.isStreaming = false;
            setUILocked(false);
            eyeHeader._setState(null);
            scrollToBottom(messagesArea);
        }

        /* ---- Clear ---- */
        function handleClear() {
            if (state.messages.length === 0) return;

            showConfirm({
                title: 'Clear conversation',
                message: 'This will remove all messages. This action cannot be undone.',
                confirmText: 'Clear',
                cancelText: 'Keep',
                variant: 'danger',
                onConfirm: () => {
                    state.messages = [];
                    messagesArea.innerHTML = '';
                    const empty = h('div', { className: 'node-chat__empty' },
                        h('p', {}, 'Ask a question or share an idea to get started…')
                    );
                    messagesArea.appendChild(empty);
                    eyeHeader._setState(null);
                    if (opts.onClear) opts.onClear();
                }
            });
        }

        /* ---- Lock/Unlock UI ---- */
        function setUILocked(locked) {
            textarea.disabled = locked;
            sendBtn.disabled = locked;
            if (!locked) textarea.focus();
        }

        /* ---- Public API ---- */
        const chatAPI = {
            /** Get current messages */
            getMessages: () => [...state.messages],
            /** Get current chat history in {role,content} format */
            getChatHistory: () => [...state.messages],
            /** Whether we're currently streaming */
            isStreaming: () => state.isStreaming,

            /** Start an AI response bubble (call before streaming chunks) */
            startAIResponse,
            /** Update the AI bubble with accumulated text */
            updateAIResponse,
            /** Mark streaming as done, unlock UI */
            finishStreaming,
            /** Add a complete AI message at once (for SSE FINAL_ANSWER) */
            appendAIMessage,
            /** Add thinking indicator manually */
            addThinking,
            /** Remove thinking indicator */
            removeThinking,
            /** Set FluxEye state: 'thinking' | 'animate' | 'pulsar' | null */
            setEyeState: (s) => eyeHeader._setState(s),
            /** Lock/unlock UI */
            setUILocked,
            /** Programmatically clear chat */
            clearChat: () => {
                state.messages = [];
                messagesArea.innerHTML = '';
                const empty = h('div', { className: 'node-chat__empty' },
                    h('p', {}, 'Ask a question or share an idea to get started…')
                );
                messagesArea.appendChild(empty);
                eyeHeader._setState(null);
            },
            /** Access the root DOM element */
            rootElement: root,
        };

        return chatAPI;
    }

    /* ---- Public module ---- */
    return { init, createFluxEye, showConfirm };
})();

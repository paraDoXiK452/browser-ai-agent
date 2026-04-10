"""Browser automation via Playwright for GPT-5.4 Computer Use.

The model sees screenshots and returns coordinate-based actions.
No DOM parsing, no element indices — pure visual interaction.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright_stealth import Stealth

# Playwright uses different key names than the model
_KEY_MAP = {
    "ENTER": "Enter", "RETURN": "Enter", "TAB": "Tab",
    "ESCAPE": "Escape", "ESC": "Escape",
    "SPACE": " ", "BACKSPACE": "Backspace", "DELETE": "Delete",
    "ARROWUP": "ArrowUp", "ARROWDOWN": "ArrowDown",
    "ARROWLEFT": "ArrowLeft", "ARROWRIGHT": "ArrowRight",
    "UP": "ArrowUp", "DOWN": "ArrowDown", "LEFT": "ArrowLeft", "RIGHT": "ArrowRight",
    "CTRL": "Control", "CONTROL": "Control",
    "ALT": "Alt", "SHIFT": "Shift", "META": "Meta",
    "HOME": "Home", "END": "End", "PAGEUP": "PageUp", "PAGEDOWN": "PageDown",
    "CAPSLOCK": "CapsLock", "NUMLOCK": "NumLock",
    "F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4", "F5": "F5",
    "F6": "F6", "F7": "F7", "F8": "F8", "F9": "F9", "F10": "F10",
    "F11": "F11", "F12": "F12",
}


def _normalize_key(key: str) -> str:
    return _KEY_MAP.get(key.upper(), key)


@dataclass
class BrowserConfig:
    headless: bool = False
    slow_mo_ms: int = 100
    viewport_width: int = 1440
    viewport_height: int = 900
    use_chrome: bool = True
    chrome_profile: str = ""
    chrome_profile_directory: str = ""
    persist_session: bool = True
    persist_profile_dir: str = ".agent_profile"
    launch_timeout_s: float = 20.0


class BrowserSession:
    def __init__(self, config: BrowserConfig):
        self.config = config
        self._pw = None
        self._browser: Optional[Browser] = None
        self._ctx: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._observed_ids: set[str] = set()
        self._observed_elements: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        # In some Windows setups Playwright startup can hang (e.g., broken driver install).
        # Time out explicitly so the CLI doesn't look "frozen".
        self._pw = await asyncio.wait_for(
            async_playwright().start(),
            timeout=max(5.0, float(self.config.launch_timeout_s)),
        )
        args = [
            "--no-first-run",
            "--no-default-browser-check",
        ]
        ignore = ["--enable-automation", "--no-sandbox", "--enable-blink-features=AutomationControlled"]
        # When use_chrome=True, use the system Chrome channel.
        channel = "chrome" if self.config.use_chrome else None

        async def _launch_non_persistent() -> None:
            self._browser = await self._pw.chromium.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo_ms,
                channel=channel,
                args=args,
                ignore_default_args=ignore,
            )
            self._ctx = await self._browser.new_context(
                viewport={"width": self.config.viewport_width, "height": self.config.viewport_height},
                locale="ru-RU",
                timezone_id="Europe/Moscow",
                ignore_https_errors=True,
            )
            self.page = await self._ctx.new_page()

        if self.config.persist_session:
            # Use a persistent profile by default so logins/cookies survive restarts.
            # If persistent launch fails (common on some Windows setups), fall back to
            # a non-persistent context instead of crashing the whole run.
            # Only use a real Chrome user-data-dir when explicitly using system Chrome.
            # Otherwise stick to the agent-owned profile directory.
            preferred_user_data_dir = (
                self.config.chrome_profile.strip()
                if (self.config.use_chrome and self.config.chrome_profile.strip())
                else self.config.persist_profile_dir
            )
            fallback_user_data_dir = self.config.persist_profile_dir

            async def _launch_persistent(user_data_dir: str) -> BrowserContext:
                profile_path = Path(user_data_dir).expanduser().resolve()
                profile_path.mkdir(parents=True, exist_ok=True)
                launch_args = list(args)
                if self.config.use_chrome and self.config.chrome_profile_directory:
                    launch_args.append(f"--profile-directory={self.config.chrome_profile_directory}")
                return await self._pw.chromium.launch_persistent_context(
                    user_data_dir=str(profile_path),
                    channel=channel,
                    headless=self.config.headless,
                    slow_mo=self.config.slow_mo_ms,
                    args=launch_args,
                    ignore_default_args=ignore,
                    viewport={"width": self.config.viewport_width, "height": self.config.viewport_height},
                    locale="ru-RU",
                    timezone_id="Europe/Moscow",
                    ignore_https_errors=True,
                )

            # If user asked for system Chrome + explicit user-data-dir, do not fall back silently
            # (fallback would log the user out). Instead, raise with a clear message.
            force_real_profile = (
                os.getenv("FORCE_CHROME_PROFILE", "0").strip() == "1"
                or (self.config.use_chrome and bool(self.config.chrome_profile.strip()))
            )
            try:
                try:
                    self._ctx = await asyncio.wait_for(
                        _launch_persistent(preferred_user_data_dir),
                        timeout=max(5.0, float(self.config.launch_timeout_s)),
                    )
                except Exception:
                    # Real Chrome profiles are frequently locked by running Chrome.
                    # Fall back to an isolated persistent profile unless forced.
                    if force_real_profile or preferred_user_data_dir == fallback_user_data_dir:
                        raise RuntimeError(
                            "Не удалось запустить системный Chrome с указанным профилем. "
                            "Обычно это значит, что профиль занят (Chrome открыт). "
                            "Полностью закройте все окна Chrome (и проверьте в Диспетчере задач, что chrome.exe не висит), "
                            "затем запустите агента снова."
                        )
                    self._ctx = await asyncio.wait_for(
                        _launch_persistent(fallback_user_data_dir),
                        timeout=max(5.0, float(self.config.launch_timeout_s)),
                    )
                self.page = self._ctx.pages[0] if self._ctx.pages else await self._ctx.new_page()
            except Exception:
                # Persistent mode failed; retry with a clean non-persistent context.
                if force_real_profile:
                    raise
                await _launch_non_persistent()
        else:
            await _launch_non_persistent()

        try:
            stealth = Stealth(
                navigator_languages_override=("ru-RU", "ru"),
                navigator_platform_override="Win32",
                navigator_vendor_override="Google Inc.",
            )
            await asyncio.wait_for(stealth.apply_stealth_async(self._ctx), timeout=5.0)
        except Exception:
            pass

        _ANTI_DETECT_JS = """() => {
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['ru-RU', 'ru', 'en-US', 'en']});
            if (window.chrome === undefined) {
                window.chrome = {runtime: {}, loadTimes: function(){}, csi: function(){}};
            }
        }"""
        try:
            await self._ctx.add_init_script(_ANTI_DETECT_JS)
            await self.page.evaluate(_ANTI_DETECT_JS)
        except Exception:
            pass

        self.page.on("dialog", lambda d: d.dismiss())

    async def close(self) -> None:
        try:
            if self.page and not self.page.is_closed():
                try:
                    await self.page.wait_for_timeout(250)
                except Exception:
                    pass
            if self._ctx:
                try:
                    await self._ctx.close()
                except Exception:
                    pass
                self._ctx = None
                self.page = None
            if self._browser:
                try:
                    await self._browser.close()
                except Exception:
                    pass
                self._browser = None
        finally:
            if self._pw:
                try:
                    await self._pw.stop()
                except Exception:
                    pass
                self._pw = None

    def _p(self) -> Page:
        if not self.page:
            raise RuntimeError("Browser not started")
        return self.page

    async def _ensure_page(self) -> Page:
        """Make sure self.page points to a live page. Recover if closed."""
        if self.page and not self.page.is_closed():
            return self.page
        if self._ctx:
            try:
                pages = [p for p in self._ctx.pages if not p.is_closed()]
                if pages:
                    self.page = pages[-1]
                    return self.page
                self.page = await self._ctx.new_page()
                return self.page
            except Exception as e:
                raise RuntimeError(
                    "Browser context is no longer available. "
                    "If you are using a real Chrome profile, close all regular Chrome windows and start the agent again."
                ) from e
        raise RuntimeError("Browser context closed")

    async def _wait_for_page_stability(self, timeout_ms: int = 5000) -> None:
        """Wait briefly for navigation/rendering to settle before the next screenshot."""
        page = await self._ensure_page()
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 2500))
        except Exception:
            pass
        try:
            await page.wait_for_function(
                "document.readyState === 'complete' || document.readyState === 'interactive'",
                timeout=timeout_ms,
            )
        except Exception:
            pass
        await page.wait_for_timeout(700)

    # ── Tab management ──

    async def close_extra_tabs(self) -> None:
        """Close extra tabs, keep the latest non-closed one."""
        if not self._ctx:
            return
        pages = [p for p in self._ctx.pages if not p.is_closed()]
        if len(pages) <= 1:
            if pages:
                self.page = pages[0]
            return
        self.page = pages[-1]
        for p in pages[:-1]:
            try:
                await p.close()
            except Exception:
                pass

    # ── High-level tools (called as custom function tools) ──

    async def navigate(self, url: str) -> str:
        p = await self._ensure_page()
        try:
            resp = await p.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await self._wait_for_page_stability()
            status = resp.status if resp else None
            return f"Navigated to {p.url} (status {status})"
        except Exception as e:
            return f"Navigation failed: {e}"

    async def search_web(self, query: str) -> str:
        p = await self._ensure_page()
        try:
            from urllib.parse import quote_plus

            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            await p.goto(search_url, wait_until="domcontentloaded", timeout=30_000)
            await self._wait_for_page_stability()
        except Exception as e:
            return f"Failed to open web search: {e}"
        try:
            title = await p.title()
        except Exception:
            title = ""
        return f"Searched the web for '{query}'. Page: {title} ({p.url})"

    async def current_url(self) -> str:
        p = await self._ensure_page()
        return p.url

    async def go_back(self) -> str:
        page = await self._ensure_page()
        try:
            resp = await page.go_back(wait_until="domcontentloaded", timeout=15_000)
            await self._wait_for_page_stability(timeout_ms=2500)
            if resp is None:
                return f"Went back. Current page: {page.url}"
            return f"Went back to {page.url} (status {resp.status})"
        except Exception as e:
            return f"Back navigation failed: {e}"

    async def cart_snapshot(self) -> str:
        """
        Best-effort cart snapshot from visible page text.
        Avoids pulling unrelated recommendation blocks by slicing the text between
        known checkout section markers.
        """
        page = await self._ensure_page()
        script = r"""() => {
            const normalizeLines = (text) => (text || '')
              .replace(/\r/g, '')
              .split('\n')
              .map((l) => l.replace(/\s+/g, ' ').trim())
              .filter(Boolean);

            const fullText = (document.body ? document.body.innerText : '') || '';
            const lines = normalizeLines(fullText);
            const joined = lines.join('\n');

            const markersStart = [
              'Cart', 'Your cart', 'Your order', 'Order summary',
              'Корзина', 'Ваш заказ', 'Ваша корзина', 'Состав заказа',
            ];
            const markersEnd = [
              'Payment', 'Pay', 'Place order', 'Total', 'Subtotal', 'Checkout',
              'Оплата', 'Оплатить', 'Итого', 'Оформить заказ',
              'Способ оплаты', 'Terms', 'Privacy', 'Footer',
              'Условия', 'О компании', 'Контакты', 'Политика конфиденциальности',
              'Пользовательское соглашение',
            ];

            const findIndexOfAny = (haystack, needles) => {
              for (const n of needles) {
                const idx = haystack.indexOf(n);
                if (idx !== -1) return {needle: n, idx};
              }
              return {needle: '', idx: -1};
            };

            // Prefer the mini-cart block: "Корзина" often appears twice (nav + sidebar).
            // lastIndexOf picks the sidebar cart on restaurant pages.
            let start = joined.lastIndexOf('Корзина');
            if (start === -1) start = joined.indexOf('Ваш заказ');
            if (start === -1) start = 0;

            let end = -1;
            if (start !== -1) {
              const after = joined.slice(start);
              const endHit = findIndexOfAny(after, markersEnd);
              end = endHit.idx === -1 ? -1 : (start + endHit.idx);
            }

            const sliceText = end !== -1 ? joined.slice(start, end) : joined.slice(start);
            let sliceLines = normalizeLines(sliceText);

            // Address heuristic from full text (usually near top of checkout page)
            let address = '';
            const addrRegexes = [
              /улиц[аы]\s+[^\n,]+,\s*\d+[^\n]*/i,
              /проспект\s+[^\n,]+,\s*\d+[^\n]*/i,
              /бульвар\s+[^\n,]+,\s*\d+[^\n]*/i,
              /шоссе\s+[^\n,]+,\s*\d+[^\n]*/i,
            ];
            for (const re of addrRegexes) {
              const m = joined.match(re);
              if (m && m[0]) { address = m[0].trim(); break; }
            }

            // Item extraction: look for name lines followed by a price/qty line.
            const items = [];
            const hasLetters = (l) => /[A-Za-zА-Яа-яЁё]/.test(l);
            const isNoise = (l) => {
              const low = l.toLowerCase();
              return (
                low === 'cart' || low === 'your order' || low === 'your cart' ||
                low === 'корзина' || low === 'ваш заказ' || low === 'ваша корзина' ||
                low.includes('clear cart') || low.includes('очистить') ||
                low.includes('delivery') || low.includes('доставка') ||
                low.includes('payment') || low.includes('оплат') ||
                low.includes('promo') || low.includes('промокод') ||
                low.includes('service fee') || low.includes('сервисн') ||
                low.includes('shipping') || low.includes('tax') ||
                low === 'total' || low === 'subtotal' ||
                low === 'итого' || low.startsWith('итого') ||
                // UI-only lines: bare prices, bare numbers, weight labels
                /^\d+\s*[₽$€£¥]?$/.test(low) ||
                /^\d+$/.test(low) ||
                /^·?\s*\d+\s*[гgmlмл]$/.test(low)
              );
            };
            const hasRub = (l) => /₽/.test(l);
            const qtyFromLine = (l) => {
              const m = l.match(/[–-]\s*(\d+)\s*\+/);
              if (m) return parseInt(m[1], 10);
              return null;
            };

            const extractFromLines = (lines) => {
              const out = [];
              for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                if (!line || isNoise(line)) continue;
                const next = lines[i + 1] || '';
                const next2 = lines[i + 2] || '';

                // Pattern A: "Name" then "… – 1 + 108 ₽"
                if (hasLetters(line) && !hasRub(line) && (hasRub(next) || hasRub(next2))) {
                  const priceLine = hasRub(next) ? next : next2;
                  const qty = qtyFromLine(priceLine) ?? qtyFromLine(next) ?? null;
                  out.push({ name: line, qty: qty ?? 1, raw: priceLine });
                  continue;
                }

                // Pattern B: single line contains both name and ₽ (e.g. "Чизбургер, 113 ₽, 117 г")
                if (hasRub(line) && !isNoise(line)) {
                  if (!hasLetters(line)) continue;
                  const qty = qtyFromLine(line);
                  let name = line
                    .replace(/\s*[–-]\s*\d+\s*\+\s*.*$/, '')
                    .replace(/\s*[,.]?\s*\d+\s*₽.*$/, '')
                    .trim() || line;
                  out.push({ name, qty: qty ?? 1, raw: line });
                }
              }
              return out;
            };

            items.push(...extractFromLines(sliceLines));

            // Wider cart window (do not scan the whole page — avoids menu/footer false positives)
            const kCart = joined.lastIndexOf('Корзина');
            const cartWindow =
              kCart === -1 ? joined : joined.slice(Math.max(0, kCart - 200), Math.min(joined.length, kCart + 4500));

            if (items.length === 0) {
              items.push(...extractFromLines(normalizeLines(cartWindow)));
            }

            // Last resort: priced lines only inside cartWindow
            if (items.length === 0) {
              const rx = /([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\s–\-—]{0,72})\s*[,.]?\s*\d+\s*₽/g;
              let m;
              while ((m = rx.exec(cartWindow)) !== null) {
                const name = (m[1] || '').replace(/\s+/g, ' ').trim();
                const low = name.toLowerCase();
                if (!name || low.length < 3) continue;
                if (isNoise(name)) continue;
                items.push({ name, qty: 1, raw: m[0] });
              }
            }

            // De-dup by name, prefer higher qty
            const merged = new Map();
            for (const it of items) {
              const key = (it.name || '').toLowerCase();
              if (!key) continue;
              const prev = merged.get(key);
              if (!prev) merged.set(key, it);
              else merged.set(key, { ...prev, qty: Math.max(prev.qty || 1, it.qty || 1) });
            }

            return {
              url: location.href,
              address,
              cart_section_text: sliceLines.slice(0, 80).join('\n'),
              items: Array.from(merged.values()).slice(0, 20),
            };
        }"""
        payload = await page.evaluate(script)
        return json.dumps(payload, ensure_ascii=False)

    def get_observed_element(self, element_id: str) -> dict[str, Any] | None:
        return self._observed_elements.get(element_id)

    def clear_observed(self) -> None:
        self._observed_ids.clear()
        self._observed_elements.clear()

    async def click_observed(self, element_id: str) -> str:
        page = await self._ensure_page()
        if element_id not in self._observed_ids:
            return f"Observed element {element_id} is not available. Run observe() again."
        selector = f'[data-agent-observe-id="{element_id}"]'
        locator = page.locator(selector).first
        if await locator.count() == 0:
            return f"Observed element {element_id} was not found. Run observe() again."
        try:
            await locator.scroll_into_view_if_needed(timeout=5_000)
        except Exception:
            pass
        label = ""
        try:
            label = await locator.evaluate(
                """(el) => (el.innerText || el.textContent || el.getAttribute('aria-label') || el.getAttribute('title') || '').replace(/\\s+/g, ' ').trim().slice(0, 160)"""
            )
        except Exception:
            pass
        try:
            await locator.click(timeout=10_000)
        except Exception:
            try:
                await locator.click(timeout=3_000, force=True)
            except Exception as exc:
                try:
                    await locator.evaluate("(el) => el.click()")
                except Exception:
                    return f"Failed to click observed element {element_id}: {exc}"
        await self._wait_for_page_stability(timeout_ms=2500)
        self.clear_observed()
        return f"Clicked observed element {element_id}: {label or '(no label)'}"

    async def submit_observed_search(self, element_id: str) -> str:
        """Focus an observed input and press Enter to commit a search/query (fallback when no button is obvious)."""
        page = await self._ensure_page()
        if element_id not in self._observed_ids:
            return f"Observed element {element_id} is not available. Run observe() again."
        selector = f'[data-agent-observe-id="{element_id}"]'
        locator = page.locator(selector).first
        if await locator.count() == 0:
            return f"Observed element {element_id} was not found. Run observe() again."
        try:
            await locator.scroll_into_view_if_needed(timeout=5_000)
        except Exception:
            pass
        label = ""
        try:
            label = await locator.evaluate(
                """(el) => (el.innerText || el.textContent || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '').replace(/\\s+/g, ' ').trim().slice(0, 160)"""
            )
        except Exception:
            pass
        try:
            await locator.click(timeout=10_000)
        except Exception:
            try:
                await locator.click(timeout=3_000, force=True)
            except Exception as exc:
                try:
                    await locator.evaluate("(el) => { el.focus(); }")
                except Exception:
                    return f"Failed to focus observed element {element_id} for submit: {exc}"
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(350)
        await self._wait_for_page_stability(timeout_ms=2500)
        self.clear_observed()
        return f"Submitted search via Enter on observed element {element_id}: {label or '(no label)'}"

    # ── Screenshot for the model ──

    async def screenshot(self) -> str:
        """Capture PNG screenshot as base64 data URL for Computer Use."""
        page = await self._ensure_page()
        try:
            await self._wait_for_page_stability(timeout_ms=2500)
        except Exception:
            pass
        try:
            data = await asyncio.wait_for(
                page.screenshot(type="png"), timeout=15.0,
            )
        except asyncio.TimeoutError:
            data = await page.screenshot(type="png", full_page=False)
        except Exception:
            page = await self._ensure_page()
            data = await page.screenshot(type="png")
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"

    async def inspect_state(self) -> str:
        page = await self._ensure_page()
        script = """() => {
            const normalize = (text) => (text || '').replace(/\\s+/g, ' ').trim();
            const bodyText = normalize(document.body ? document.body.innerText : '');
            const title = normalize(document.title || '');
            const dialogs = Array.from(document.querySelectorAll('[role="dialog"], dialog, [aria-modal="true"]'));
            const visibleDialogs = dialogs.filter((el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 40 && rect.height > 40;
            });
            const dialogTexts = visibleDialogs.map((el) => normalize(el.innerText).slice(0, 300));
            const blockerKeywords = ['captcha', 'капча', 'я не робот', 'i am not a robot', 'showcaptcha', 'recaptcha'];
            const cookieKeywords = ['cookie', 'cookies', 'файл cookie', 'куки', 'allow all', 'accept all', 'принять'];
            const addressKeywords = [
                'адрес', 'улица', 'дом', 'квартира', 'подъезд', 'доставка', 'выбрать улицу',
                'куда доставить заказ', 'укажите адрес', 'введите адрес', 'адрес доставки',
                'квартира или офис', 'подтвердите адрес', 'доставить сюда'
            ];
            const cartKeywords = ['корзина', 'оформить заказ', 'checkout'];
            const overlays = Array.from(document.querySelectorAll('body *')).filter((el) => {
                const style = window.getComputedStyle(el);
                if (style.position !== 'fixed' && style.position !== 'sticky') return false;
                const rect = el.getBoundingClientRect();
                if (rect.width < window.innerWidth * 0.25 || rect.height < 40) return false;
                if (style.display === 'none' || style.visibility === 'hidden') return false;
                return true;
            }).slice(0, 8).map((el) => normalize(el.innerText).slice(0, 220));
            const modalAndOverlayText = `${dialogTexts.join(' ')} ${overlays.join(' ')}`.toLowerCase();
            const textLower = `${title} ${bodyText} ${modalAndOverlayText}`.toLowerCase();
            return {
                url: location.href,
                title,
                body_text: bodyText.slice(0, 5000),
                dialog_texts: dialogTexts,
                overlay_texts: overlays,
                flags: {
                    captcha: blockerKeywords.some((word) => textLower.includes(word)),
                    cookie_banner: cookieKeywords.some((word) => textLower.includes(word)),
                    address_modal: addressKeywords.some((word) => modalAndOverlayText.includes(word)),
                    cart_visible: cartKeywords.some((word) => textLower.includes(word)),
                    modal_visible: visibleDialogs.length > 0,
                }
            };
        }"""
        payload = await page.evaluate(script)
        return json.dumps(payload, ensure_ascii=False)

    async def dismiss_blockers(self) -> str:
        page = await self._ensure_page()
        script = """() => {
            const normalize = (text) => (text || '').replace(/\\s+/g, ' ').trim();
            const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 10 && rect.height > 10;
            };
            const labels = [
                'allow all', 'accept all', 'accept', 'ok', 'close', 'dismiss',
                'принять', 'разрешить', 'закрыть', 'не сейчас'
            ];
            const visibleDialogs = Array.from(document.querySelectorAll('[role="dialog"], dialog, [aria-modal="true"]')).filter(isVisible);
            const roots = visibleDialogs.length ? visibleDialogs : [document.body];
            const candidates = roots.flatMap((root) => Array.from(root.querySelectorAll('button, [role="button"], [aria-label]')));
            for (const el of candidates) {
                if (!isVisible(el)) continue;
                const rect = el.getBoundingClientRect();
                if (rect.width > 300 || rect.height > 100) continue;
                const label = normalize(el.innerText || el.textContent || el.getAttribute('aria-label') || el.getAttribute('title')).toLowerCase();
                if (!label || label.length > 40) continue;
                if (labels.some((item) => label === item)) {
                    el.click();
                    return {action: 'clicked', label};
                }
            }
            return {action: 'none'};
        }"""
        outcome = await page.evaluate(script)
        await self._wait_for_page_stability(timeout_ms=1500)
        action = outcome.get("action", "none")
        label = outcome.get("label", "")
        if action == "clicked":
            return f"Dismissed blocker via visible control: {label}"
        return "No dismissible blocker found"

    async def observe(self, goal: str = "") -> str:
        page = await self._ensure_page()
        script = """(goalText) => {
            const previous = document.querySelectorAll('[data-agent-observe-id]');
            previous.forEach((el) => el.removeAttribute('data-agent-observe-id'));

            const selectors = [
              'button', 'a', 'input', 'textarea', 'select',
              '[role="button"]', '[role="link"]', '[role="menuitem"]', '[role="option"]',
              '[role="tab"]', '[role="checkbox"]', '[role="radio"]', '[role="combobox"]',
              '[role="searchbox"]', '[role="textbox"]', '[contenteditable="true"]', '[onclick]'
            ];

            const isVisible = (el) => {
              const style = window.getComputedStyle(el);
              if (style.visibility === 'hidden' || style.display === 'none') return false;
              const rect = el.getBoundingClientRect();
              if (rect.width < 8 || rect.height < 8) return false;
              if (rect.bottom < 0 || rect.right < 0) return false;
              if (rect.top > window.innerHeight || rect.left > window.innerWidth) return false;
              return true;
            };

            const getLabel = (el) => {
              const parts = [
                el.getAttribute('aria-label') || '',
                el.getAttribute('placeholder') || '',
                el.getAttribute('title') || '',
                el.getAttribute('alt') || '',
                el.getAttribute('name') || '',
                el.getAttribute('type') || '',
                el.value || '',
                el.innerText || el.textContent || ''
              ];
              return parts.join(' ').replace(/\\s+/g, ' ').trim();
            };

            const centerUnobstructed = (el, rect) => {
              const cx = Math.min(window.innerWidth - 1, Math.max(0, rect.left + rect.width / 2));
              const cy = Math.min(window.innerHeight - 1, Math.max(0, rect.top + rect.height / 2));
              const topEl = document.elementFromPoint(cx, cy);
              return !!topEl && (topEl === el || el.contains(topEl) || topEl.contains(el));
            };

            const goal = (goalText || '').toLowerCase();
            const items = [];
            const seen = new Set();
            let counter = 1;
            const visibleDialogs = Array.from(document.querySelectorAll('[role="dialog"], dialog, [aria-modal="true"]')).filter(isVisible);
            const roots = visibleDialogs.length ? visibleDialogs : [document];

            for (const root of roots) {
              for (const el of root.querySelectorAll(selectors.join(','))) {
                if (!isVisible(el)) continue;
                const rect = el.getBoundingClientRect();
                if (!centerUnobstructed(el, rect)) continue;
                const label = getLabel(el);
                const role = (el.getAttribute('role') || el.tagName || '').toLowerCase();
                const key = [role, label, Math.round(rect.left), Math.round(rect.top)].join('|');
                if (seen.has(key)) continue;
                seen.add(key);

                const id = String(counter++);
                el.setAttribute('data-agent-observe-id', id);

                const haystack = `${role} ${label}`.toLowerCase();
                let score = 0;
                if (goal && haystack.includes(goal)) score += 100;
                if (goal) {
                  for (const token of goal.split(/\\s+/).filter(Boolean)) {
                    if (haystack.includes(token)) score += 10;
                  }
                }
                if (visibleDialogs.length) score += 20;
                if (role.includes('searchbox') || label.toLowerCase().includes('поиск') || label.toLowerCase().includes('найти')) score += 2;
                if (role.includes('button') || el.tagName === 'BUTTON') score += 3;
                if (role.includes('link') || el.tagName === 'A') score += 2;
                if (el.tagName === 'INPUT') score += 1;

                items.push({
                  id,
                  role,
                  label: label.slice(0, 160),
                  x: Math.round(rect.left + rect.width / 2),
                  y: Math.round(rect.top + rect.height / 2),
                  width: Math.round(rect.width),
                  height: Math.round(rect.height),
                  score
                });
              }
            }

            items.sort((a, b) => {
              if (b.score !== a.score) return b.score - a.score;
              if (a.y !== b.y) return a.y - b.y;
              return a.x - b.x;
            });
            return items.slice(0, 60);
        }"""
        observed = await page.evaluate(script, goal)
        self._observed_ids = {item["id"] for item in observed}
        self._observed_elements = {item["id"]: item for item in observed}
        payload = {
            "url": page.url,
            "goal": goal,
            "elements": observed,
        }
        return json.dumps(payload, ensure_ascii=False)

    async def type_into_observed(self, element_id: str, text: str, replace: bool = True) -> str:
        page = await self._ensure_page()
        if element_id not in self._observed_ids:
            return f"Observed element {element_id} is not available. Run observe() again."
        selector = f'[data-agent-observe-id="{element_id}"]'
        locator = page.locator(selector).first
        if await locator.count() == 0:
            return f"Observed element {element_id} was not found. Run observe() again."
        try:
            await locator.scroll_into_view_if_needed(timeout=5_000)
        except Exception:
            pass
        label = ""
        try:
            label = await locator.evaluate(
                """(el) => (el.innerText || el.textContent || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '').replace(/\\s+/g, ' ').trim().slice(0, 160)"""
            )
        except Exception:
            pass
        try:
            await locator.click(timeout=10_000)
        except Exception:
            try:
                await locator.click(timeout=3_000, force=True)
            except Exception as exc:
                try:
                    await locator.evaluate("(el) => { el.focus(); el.click && el.click(); }")
                except Exception:
                    return f"Failed to focus observed element {element_id}: {exc}"
        typed_via_fill = False
        try:
            tag = await locator.evaluate("(el) => (el.tagName || '').toLowerCase()")
        except Exception:
            tag = ""
        try:
            if tag in {"input", "textarea"}:
                if replace:
                    await locator.fill(text, timeout=5_000)
                else:
                    await locator.evaluate(
                        "(el, value) => { el.focus(); el.value = (el.value || '') + value; el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); }",
                        text,
                    )
                typed_via_fill = True
        except Exception:
            typed_via_fill = False
        if not typed_via_fill:
            if replace:
                await page.keyboard.press("Control+A")
            await page.keyboard.type(text, delay=20)
        await page.wait_for_timeout(250)
        await self._wait_for_page_stability(timeout_ms=1500)
        return f"Typed into observed element {element_id}: {label or '(no label)'}"

    # ── Execute Computer Use actions ──

    async def execute_action(self, action: Any) -> None:
        """Execute a single action from the Computer Use API response."""
        p = await self._ensure_page()
        action_type = action.type

        if action_type == "click":
            button = getattr(action, "button", "left")
            keys = getattr(action, "keys", None) or []
            modifiers = [_normalize_key(k) for k in keys]
            for m in modifiers:
                await p.keyboard.down(m)
            await p.mouse.click(action.x, action.y, button=button)
            for m in reversed(modifiers):
                await p.keyboard.up(m)
            await self._wait_for_page_stability(timeout_ms=2000)

        elif action_type == "double_click":
            await p.mouse.dblclick(action.x, action.y)
            await self._wait_for_page_stability(timeout_ms=2000)

        elif action_type == "type":
            await p.keyboard.type(action.text, delay=20)
            await p.wait_for_timeout(250)

        elif action_type == "keypress":
            keys = action.keys
            combo = "+".join(_normalize_key(k) for k in keys)
            await p.keyboard.press(combo)
            await self._wait_for_page_stability(timeout_ms=2000)

        elif action_type == "scroll":
            x = getattr(action, "x", 640)
            y = getattr(action, "y", 450)
            scroll_x = getattr(action, "scroll_x", 0)
            scroll_y = getattr(action, "scroll_y", 0)
            await p.mouse.move(x, y)
            await p.mouse.wheel(scroll_x, scroll_y)
            await p.wait_for_timeout(300)

        elif action_type == "drag":
            path = action.path
            if path and len(path) >= 2:
                await p.mouse.move(path[0]["x"], path[0]["y"])
                await p.mouse.down()
                for point in path[1:]:
                    await p.mouse.move(point["x"], point["y"])
                await p.mouse.up()

        elif action_type == "move":
            await p.mouse.move(action.x, action.y)

        elif action_type == "wait":
            await self._wait_for_page_stability(timeout_ms=2500)

        elif action_type == "screenshot":
            pass  # handled by the caller after all actions

    async def execute_actions(self, actions: list[Any]) -> None:
        """Execute a batch of actions from a computer_call."""
        for i, action in enumerate(actions):
            if action.type == "type" and i > 0:
                await self._p().wait_for_timeout(300)
            await self.execute_action(action)
            await self._p().wait_for_timeout(150)

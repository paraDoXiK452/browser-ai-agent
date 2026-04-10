"""Lightweight local eval harness for runtime policies."""

from __future__ import annotations

from agent.policy import (
    EvaluationResult,
    build_task_profile,
    classify_dead_end,
    address_tokens_visible,
    extract_address_tokens,
    extract_requested_entities,
    extract_site_query,
    extract_target_restaurant,
    infer_task_domains,
    infer_task_kind,
    infer_page_mode,
    infer_search_scope,
    is_address_commit_control,
    is_address_like_field,
    is_authorization_request,
    is_probable_restaurant_card,
    is_search_commit_control_click,
    is_search_like_field,
    parse_evaluation,
    text_matches_target,
    should_soft_accept_address,
    verify_task_completion,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_policy_evals() -> None:
    parsed = parse_evaluation(
        '{"status":"FIX","checkpoint_state":"DRIFTED","flags":["dead_end","wrong_search_context"],'
        '"evidence":"Виден футер и пустой результат.","correction":"Вернуться назад."}'
    )
    _assert(parsed.status == "FIX", "status parsing failed")
    _assert(parsed.checkpoint_state == "DRIFTED", "checkpoint_state parsing failed")
    _assert(parsed.has_flag("dead_end"), "dead_end flag missing")
    _assert(classify_dead_end(parsed), "dead_end classification failed")

    address_result = EvaluationResult(
        status="OK",
        checkpoint_state="PENDING",
        flags=[],
        evidence="Адрес сверху: улица Белоглазова, 27, подъезд 1.",
        correction="",
        raw="Адрес сверху: улица Белоглазова, 27, подъезд 1.",
    )
    _assert(
        should_soft_accept_address(
            checkpoint_text="Подтвердить адрес доставки",
            current_url="https://eda.yandex.ru/Almetyevsk?shippingType=delivery",
            result=address_result,
        ),
        "soft address accept failed",
    )

    wrong_result = EvaluationResult(
        status="FIX",
        checkpoint_state="DRIFTED",
        flags=["wrong_item"],
        evidence="Открыт не тот товар.",
        correction="Вернуться и выбрать нужный товар.",
        raw="Открыт не тот товар.",
    )
    _assert(not classify_dead_end(wrong_result), "wrong item should not classify as dead end")

    _assert(
        infer_task_domains("Закажи мне бургер в Яндекс Еде") == set(),
        "task domain inference should return empty set for non-URL service names",
    )
    _assert(
        infer_task_domains("Открой hh.ru и найди вакансии") == {"hh.ru"},
        "task domain inference should extract explicit domains",
    )
    _assert(infer_task_kind("Прочитай 10 писем и удали спам") == "mail", "task kind inference failed")
    _assert(
        extract_target_restaurant(
            "В яндекс еде закажи мне чикенбургер и большую колу с вкусно и точка, добавь в корзину но не оплачивай заказ",
            task_kind="delivery",
        ) == "вкусно и точка",
        "target restaurant extraction failed",
    )
    _assert(
        extract_requested_entities(
            "В яндекс еде закажи мне чикенбургер и большую колу с вкусно и точка, добавь в корзину но не оплачивай заказ",
            task_kind="delivery",
        ) == ["чикенбургер", "большую колу"],
        "requested entity extraction failed",
    )
    _assert(
        extract_target_restaurant(
            "Зайди в яндекс еду и закажи во вкусно и точка 1 чизбургер",
            task_kind="delivery",
        ) == "вкусно и точка",
        "target restaurant extraction failed for 'во вкусно и точка'",
    )
    _assert(
        extract_requested_entities(
            "Зайди в яндекс еду и закажи во вкусно и точка 1 чизбургер, добавь в корзину но не оплачивай товар",
            task_kind="delivery",
        ) == ["чизбургер"],
        "requested entity extraction should not include restaurant name",
    )
    profile = build_task_profile("В яндекс еде закажи мне чикенбургер и большую колу с вкусно и точка")
    _assert(profile.restaurant == "вкусно и точка", "task profile restaurant failed")
    _assert(infer_page_mode(current_url="https://eda.yandex.ru/r/vkusno", page_text="Найти в ресторане", flags={}) == "unknown", "generic page mode should return unknown")
    _assert(infer_page_mode(current_url="https://example.com/cart", page_text="Оформить заказ", flags={}) == "cart", "page mode cart detection failed")
    _assert(infer_search_scope(label="Найти в ресторане", current_url="https://eda.yandex.ru/r/vkusno", page_mode="unknown") == "search", "search scope should detect search field generically")
    _assert(is_address_like_field("Куда доставить заказ", "textbox"), "address field detection failed")
    _assert(is_address_commit_control("Выбрать", "button"), "address commit control detection failed")
    _assert(is_probable_restaurant_card("Вкусно и точка 4.5 40-50 мин"), "restaurant card detection failed")
    _assert(text_matches_target("Вкусно — и точка 4.5", "вкусно и точка"), "target text match failed")
    _assert(is_authorization_request("Можно искать и открыть Вкусно и точка в Яндекс Еде?"), "authorization request detection failed")
    delivery_result = EvaluationResult(
        status="OK",
        checkpoint_state="COMPLETE",
        flags=["ready_to_finish", "cart_verified"],
        evidence="В корзине видны обе позиции.",
        correction="",
        raw='{"status":"OK","checkpoint_state":"COMPLETE","flags":["ready_to_finish","cart_verified"]}',
    )
    ok, reason = verify_task_completion(
        task="В яндекс еде закажи мне чикенбургер и большую колу с вкусно и точка, добавь в корзину но не оплачивай заказ",
        task_kind="delivery",
        current_url="https://eda.yandex.ru/cart",
        page_text="Корзина Чикенбургер 1 Большая Кола 1 Оформить заказ",
        current_checkpoint="Проверить корзину и остановиться до оплаты",
        result=delivery_result,
    )
    _assert(ok, f"delivery completion verification failed: {reason}")

    _assert(is_search_like_field("Поиск по сайту", "searchbox"), "search-like field detection failed")
    _assert(is_search_commit_control_click("Найти", "button"), "search commit button heuristic failed")
    uncommitted = verify_task_completion(
        task="test",
        task_kind="generic",
        current_url="https://example.com",
        page_text="ok",
        current_checkpoint="",
        result=delivery_result,
        has_uncommitted_search=True,
    )
    _assert(not uncommitted[0], "uncommitted search should block verify_task_completion")
    current_addr_profile = build_task_profile("В яндекс еде закажи мне чизбургер с вкусно и точка на адрес который указан, добавь в корзину товар но не оплачивай")
    _assert(current_addr_profile.restaurant == "вкусно и точка", "current-address task restaurant extraction failed")
    single_item_ok, single_item_reason = verify_task_completion(
        task="В яндекс еде закажи мне чизбургер с вкусно и точка на адрес который указан, добавь в корзину товар но не оплачивай",
        task_kind="delivery",
        current_url="https://eda.yandex.ru/r/vkusnoitochka",
        page_text="Корзина Чизбургер 1 Далее улица Белоглазова, 27",
        current_checkpoint="Проверить корзину и остановиться до оплаты",
        result=delivery_result,
    )
    _assert(single_item_ok, f"single item completion verification failed: {single_item_reason}")

    _assert(
        extract_site_query("В яндекс еде закажи мне чизбургер с вкусно и точка на адрес который указан") == "яндекс еде",
        "site query extraction failed for delivery task",
    )
    _assert(
        extract_site_query("На hh.ru найди вакансии python разработчика") == "hh.ru",
        "site query extraction failed for job task",
    )
    _assert(
        extract_site_query("В яндекс почте прочитай все новые письма") == "яндекс почте",
        "site query extraction failed for mail task",
    )
    _assert(
        len(extract_site_query("Сделай что-нибудь интересное")) <= 80,
        "site query fallback should truncate to max 80 chars",
    )

    addr_tokens = extract_address_tokens("В яндекс еда закажи мне чизбургер на адрес белоглазова 27")
    _assert("белоглазова" in addr_tokens and "27" in addr_tokens, f"address token extraction failed: {addr_tokens}")
    _assert(
        address_tokens_visible(
            "закажи на адрес белоглазова 27",
            "Яндекс Еда улица Белоглазова, 27 Доставка",
        ),
        "address_tokens_visible should detect address on page",
    )
    _assert(
        not address_tokens_visible(
            "закажи на адрес белоглазова 27",
            "Яндекс Еда Главная страница Рестораны",
        ),
        "address_tokens_visible should return False when address is not on page",
    )


if __name__ == "__main__":
    run_policy_evals()
    print("policy evals: ok")

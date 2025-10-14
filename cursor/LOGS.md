2025-10-10 11:18:33,380 - root - INFO - Frame analysis saved to 'data/analysis_reports/frame_analysis_20251010_111833.json'.

End of turn in receive audio  1760084321.4978065
Removed audio from queue 1760084321.498385
Removed audio from queue 1760084321.4986804
Removed audio from queue 1760084321.4988663
Removed audio from queue 1760084321.4990475
Removed audio from queue 1760084321.499153
Removed audio from queue 1760084321.4993372
Removed audio from queue 1760084321.499497
Removed audio from queue 1760084321.499754
Removed audio from queue 1760084321.4999366
Removed audio from queue 1760084321.5000803
Removed audio from queue 1760084321.5002503

End of turn in receive audio  1760084335.4260943
2025-10-10 11:19:04,441 - logger - ERROR - Failed to receive message: Connection was closed.
2025-10-10 11:19:04,444 - logger - ERROR - Failed to send message: Connection was closed. received 1011 (internal error) Deadline expired before operation could complete.; then sent 1011 (internal error) Deadline expired before operation could complete.
C:\Python313\Lib\ssl.py:524: UserWarning: Bad certificate in Windows certificate store: not enough data: cadata does not contain a certificate (_ssl.c:4205)
  warnings.warn(f"Bad certificate in Windows certificate store: {exc!s}")
2025-10-10 11:19:04,882 - logger - ERROR - Failed to receive message: Connection was closed.
2025-10-10 11:19:04,883 - root - ERROR - An error occurred:
  + Exception Group Traceback (most recent call last):
  |   File "D:\Projects\!Python\CursorProjects\cogamer\cogamer.py", line 639, in run
  |     async with asyncio.TaskGroup() as tg:
  |                ~~~~~~~~~~~~~~~~~^^
  |   File "C:\Python313\Lib\asyncio\taskgroups.py", line 71, in __aexit__
  |     return await self._aexit(et, exc)
  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "C:\Python313\Lib\asyncio\taskgroups.py", line 173, in _aexit
  |     raise BaseExceptionGroup(
  |     ...<2 lines>...
  |     ) from None
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)  
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "D:\Projects\!Python\CursorProjects\cogamer\ws_client.py", line 133, in receive
    |     message = await self._connection.recv()
    |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "D:\Projects\!Python\CursorProjects\cogamer\venv\Lib\site-packages\websockets\asyncio\connection.py", line 313, in recv
    |     raise self.protocol.close_exc from self.recv_exc
    | websockets.exceptions.ConnectionClosedError: received 1011 (internal error) Deadline expired before operation could complete.; then sent 1011 (internal error) Deadline expired before operation could complete.
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "D:\Projects\!Python\CursorProjects\cogamer\ws_client.py", line 143, in force_receive
    |     message = await self.receive()
    |               ^^^^^^^^^^^^^^^^^^^^
    |   File "D:\Projects\!Python\CursorProjects\cogamer\ws_client.py", line 139, in receive
    |     raise self.WebSocketConnectionClosed("Failed to receive message: Connection was closed.")
    | ws_client.WebSocketClient.WebSocketConnectionClosed: Failed to receive message: Connection was closed.
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "D:\Projects\!Python\CursorProjects\cogamer\cogamer.py", line 495, in receive_audio
    |     raw_response = await self.ws_client.force_receive()
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     message = await self.receive()
    |               ^^^^^^^^^^^^^^^^^^^^
    |   File "D:\Projects\!Python\CursorProjects\cogamer\ws_client.py", line 139, in receive
    |     raise self.WebSocketConnectionClosed("Failed to receive message: Connection was closed.")
    | ws_client.WebSocketClient.WebSocketConnectionClosed: Failed to receive message: Connection was closed.
    +------------------------------------
2025-10-10 11:19:05,272 - logger - INFO - WebSocket connection has been disconnected.



---

## 📝 Changelog: Система сохранения контекста диалога

**Дата:** 14 октября 2025  
**Версия:** 1.1.0  
**Разработчик:** AI Assistant (Cursor)

### ✅ Выполненные задачи

#### Проблема
При автоматическом переподключении каждые ~7 минут ассистент терял весь контекст диалога и начинал общение "с чистого листа", что нарушало преемственность разговора с игроком.

#### Решение
Реализована автоматическая система сохранения и восстановления контекста диалога между переподключениями.

### 🔧 Технические изменения

#### 1. Класс `GlobalContext` (`cogamer.py`)

**Добавлены новые методы:**

- `get_recent_conversation(num_pairs=3)` — извлечение последних N пар реплик
- `save_conversation_context(filepath)` — сохранение контекста в JSON файл
- `load_conversation_context(filepath)` — загрузка контекста из JSON файла

**Функциональность:**
- Сохраняет последние 3 пары реплик (игрок + ассистент)
- Автоматически создает директорию `data/` если не существует
- Обрабатывает ошибки чтения/записи без прерывания работы
- Использует кодировку UTF-8 для поддержки кириллицы
- Добавляет временные метки сохранения

#### 2. Метод `Agent.startup()` (`cogamer.py`)

**Изменения:**
- Загрузка сохраненного контекста при инициализации
- Формирование расширенной системной инструкции с историей диалога
- Передача контекста в Gemini через system_instruction

**Формат контекста в инструкции:**
```
--- КОНТЕКСТ ПРЕДЫДУЩЕГО ДИАЛОГА ---
Вот последние реплики из нашего предыдущего разговора:

Игрок: [текст]
Ассистент: [ответ]
...

Продолжай диалог естественно, учитывая этот контекст.
--- КОНЕЦ КОНТЕКСТА ---
```

#### 3. Метод `Agent.receive_audio()` (`cogamer.py`)

**Добавлено:**
- Извлечение текстовых ответов ассистента из потока
- Автоматическое добавление ответов в историю диалога
- Сохранение контекста перед переподключением
- Логирование всех операций с контекстом

**Логика работы:**
1. Получение ответа от Gemini
2. Извлечение текстовой части (если есть)
3. Сохранение в историю: `global_context.add_message("assistant", text)`
4. При достижении интервала переподключения (7 минут):
   - Сохранение контекста в файл
   - Переподключение к WebSocket
   - Загрузка контекста при инициализации

### 📁 Новые файлы

1. **`data/conversation_context.json`** — хранилище контекста диалога
   - Создается автоматически при первом сохранении
   - Содержит: recent_conversation, game, category, timestamp
   
2. **`cursor/CONTEXT_SAVE_SYSTEM.md`** — подробная документация
   - Описание проблемы и решения
   - Технические детали реализации
   - Примеры использования
   - Сценарии работы в реальном времени
   
3. **`cursor/QUICK_START_CONTEXT.md`** — краткая инструкция
   - Быстрый старт для пользователей
   - Мониторинг работы системы
   - Решение типичных проблем
   
4. **`cursor/PROMPTS,md`** — обновлена история разработки
   - Добавлена запись о реализации системы
   - Описание решения и ссылки на документацию

### 🎯 Результат

✅ **Автоматическое сохранение** последних 6 реплик перед переподключением  
✅ **Автоматическая загрузка** контекста при новом запуске  
✅ **Бесшовное переподключение** — игрок не замечает технических перезапусков  
✅ **Сохранение преемственности** диалога между сессиями  
✅ **Полное логирование** всех операций для отладки  

### 📊 Метрики

- **Количество сохраняемых реплик:** 6 (3 пары)
- **Интервал переподключения:** 7 минут (420 секунд)
- **Формат хранения:** JSON (UTF-8)
- **Путь к файлу:** `data/conversation_context.json`
- **Обработка ошибок:** Да (с логированием)

### 🔍 Логирование

**Новые сообщения в логах:**

```
INFO - Контекст диалога сохранен в 'data/conversation_context.json'
INFO - Контекст диалога загружен из 'data/conversation_context.json'
INFO - Восстановлено N реплик из предыдущего диалога
INFO - Ответ ассистента добавлен в историю: [первые 50 символов]
INFO - Время переподключения. Сохраняем контекст диалога...
INFO - Переподключение завершено с восстановлением контекста
```

### ⚙️ Настройки

**Параметры по умолчанию:**
- `num_pairs=3` — количество пар реплик для сохранения
- `filepath="data/conversation_context.json"` — путь к файлу контекста
- `RECONNECTION_INTERVAL=60*7` — интервал переподключения (7 минут)

**Легко изменяются** в коде для кастомизации под конкретные нужды.

### 🐛 Известные ограничения

- Сохраняются только текстовые реплики (голосовые преобразуются в текст если доступны)
- Максимум 3 пары реплик (можно изменить в настройках)
- Требуется доступ на запись в папку `data/`

### 🚀 Следующие шаги

**Возможные улучшения в будущем:**
- [ ] Сохранение анализа кадров в контекст
- [ ] Сжатие старых реплик для экономии места
- [ ] Настройка через конфигурационный файл
- [ ] Облачное хранение контекста
- [ ] История всех сессий для анализа

---

**Статус:** ✅ Готово к использованию  
**Тестирование:** Требуется тестирование в боевых условиях  
**Документация:** ✅ Полная  


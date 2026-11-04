import re
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9а-яё_-]{1,50}$")
MAX_DESCRIPTION = 120
MAX_CONTENT = 2000
MAX_NOTES = 100


class MemoryError(Exception):
    pass


class Memory:
    """Заметки в виде файлов: их можно читать и править руками."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        name = name.strip().lower()
        if not NAME_RE.fullmatch(name):
            raise MemoryError("имя заметки: только буквы, цифры, _ и -, до 50 символов")
        return self.directory / f"{name}.md"

    def _parse(self, path: Path) -> tuple[str, str, str]:
        head, _, content = path.read_text(encoding="utf-8").partition("\n---\n")
        fields = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
        return fields.get("description", ""), fields.get("author", ""), content

    def author(self, name: str) -> str | None:
        path = self._path(name)
        return self._parse(path)[1] if path.exists() else None

    def write(self, name: str, description: str, content: str, author: str) -> None:
        path = self._path(name)
        description = " ".join(description.split())
        if not description or len(description) > MAX_DESCRIPTION:
            raise MemoryError(f"описание: одна строка до {MAX_DESCRIPTION} символов")
        if len(content) > MAX_CONTENT:
            raise MemoryError(f"текст заметки: до {MAX_CONTENT} символов")
        if not path.exists() and len(list(self.directory.glob("*.md"))) >= MAX_NOTES:
            raise MemoryError(f"память заполнена ({MAX_NOTES} заметок), перезапиши старую заметку")
        author = " ".join(author.split())
        path.write_text(
            f"description: {description}\nauthor: {author}\n---\n{content}", encoding="utf-8"
        )

    def read(self, name: str) -> str:
        path = self._path(name)
        if not path.exists():
            raise MemoryError(f"заметки {name} нет")
        return self._parse(path)[2]

    def delete(self, name: str) -> None:
        path = self._path(name)
        if not path.exists():
            raise MemoryError(f"заметки {name} нет")
        path.unlink()

    def index(self) -> str:
        lines = []
        for path in sorted(self.directory.glob("*.md")):
            description, author, _ = self._parse(path)
            lines.append(f"- {path.stem}: {description} (автор: {author})")
        return "\n".join(lines) or "Заметок пока нет."

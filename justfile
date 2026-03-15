install:
    uv sync

test *args:
    uv run pytest {{args}}

lint:
    uv run ruff check

fmt:
    uv run ruff format

run *args:
    uv run crowd-control {{args}}

build:
    uv build

publish-test: build
    uv publish --publish-url https://test.pypi.org/legacy/ --token $(op read "op://Private/Test PyPI/Entire Account")

publish-prod: build
    uv publish --token $(op read "op://Private/PyPI/Entire Account")

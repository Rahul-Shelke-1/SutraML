### Workflow Structure

```
.github/
└── workflows/
    ├── ci.yml
    ├── release-please.yml
    └── publish.yml
```

`ci.yml`

Responsible for:

- dependency installation
- linting
- formatting checks
- unit tests
- integration tests
- package build
- possibly Python-version matrix

`release-please.yml`

Responsible for:

- analyzing conventional commits
- calculating the next version
- updating changelog
- updating package version
- creating/updating the Release PR
- creating the GitHub release/tag after the Release PR is merged

`publish.yml`

Responsible for:

- building the package
- publishing to PyPI
- potentially publishing Docker images
- other release artifacts

---

### Flow

1st flow to follow:

```mermaid
flowchart LR

A[PR] --> B[CI]
B --> C[merge]
C --> D[CI on main]
D --> E[release-please]
E --> F[Release PR]
```

Then:

```mermaid
flowchart LR

A[Release PR merged] --> B[Github Release created]
B --> C[publish.yml]
C --> D[PyPI]
```
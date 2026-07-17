# Review and Public-Release Workflow

[English](REVIEW_RELEASE_WORKFLOW.md) | [Simplified Chinese](REVIEW_RELEASE_WORKFLOW.zh-CN.md)

## What Should Be Uploaded?

Do **not** upload the author-maintained `EDA_reproducibility_package/` directory
directly to an anonymous service. That directory is the future public release
and intentionally contains author metadata in `LICENSE`, `CITATION.cff`,
`pyproject.toml`, and third-party notices.

Generate the reviewer artifacts first:

```bash
python tools/build_anonymous_snapshot.py --force
```

The command creates:

```text
dist/
|-- eda-review-anonymous/             # contents for a dedicated review repo
|-- eda-review-anonymous.zip          # Editorial Manager supplement
|-- eda-review-anonymous.zip.sha256   # immutable checksum
`-- UPLOAD_INSTRUCTIONS.txt
```

Use the **contents of `dist/eda-review-anonymous/`** for an anonymous repository.
Use `dist/eda-review-anonymous.zip` for a supplementary-code upload. The builder
removes datasets, checkpoints, outputs, provider responses, API credentials,
author metadata, and personal paths, then creates reviewer quick-start files and
scans the result for identity leaks.

## Recommended Anonymous Read-only Route

A private repository accessible only to the authors is not sufficient reviewer
access. The recommended review-stage arrangement is:

1. Create a new, empty **private GitHub repository** dedicated to the anonymous
   snapshot. Do not initialize it with another README, license, or `.gitignore`.
2. Commit only the contents of `dist/eda-review-anonymous/`. Do not copy its
   contents into an identity-bearing repository or preserve Git history from the
   authors' working repository.
3. Push one frozen commit, for example `review-v1`. Record its commit SHA and the
   ZIP SHA-256 in the submission notes.
4. Create a read-only mirror at [Anonymous GitHub](https://anonymous.4open.science/)
   from that private repository. Prefer a fixed commit URL and disable automatic
   updates during active review.
5. Add custom redaction terms for all author names, affiliations, email
   addresses, GitHub user/organization names, private repository names, and any
   distinctive project paths. Do not include those terms in the anonymous
   repository itself.
6. Disable or manually inspect external links and binary assets that could reveal
   identity. This snapshot intentionally excludes paper PDFs, generated figures,
   datasets, checkpoints, and provider responses.
7. Set the mirror expiration after the expected review period and choose removal,
   not redirection to an identity-bearing repository, while review is active.
8. Open the final anonymous URL in a signed-out browser, download its archive,
   and run `python tools/check_release.py` from the downloaded copy.

Anonymous GitHub creates a read-only identity-stripped mirror rather than making
the source GitHub repository public. Accessing a private source repository
requires the service's GitHub authorization. Review its current permissions and
retention terms before authorizing it. If that authorization is unacceptable,
upload the generated ZIP as supplementary code instead.

For a repository larger than the service's direct-download limit, use its
proxy/stream mode or reduce nonessential assets. The generated ZIP remains the
immutable fallback and should be retained with its checksum.

## Dedicated Review Repository

From the generated anonymous directory, initialize a fresh repository:

```bash
cd dist/eda-review-anonymous
git init
git branch -M main
git add .
git status --short
git commit -m "Anonymous review snapshot review-v1"
git remote add origin https://github.com/OWNER/PRIVATE_REVIEW_REPOSITORY.git
git push -u origin main
git rev-parse HEAD
```

Do not run these commands from `EDA_reproducibility_package/` when creating the
anonymous reviewer repository. That root is suitable only for the authors'
private development repository or the post-acceptance public release.

## Submission Checklist

Before submitting the revision:

1. Confirm that `REVIEWER_QUICKSTART.md` opens correctly and contains runnable
   commands from the repository root.
2. Confirm that no data images, weights, adversarial outputs, VLM raw responses,
   secrets, personal paths, author names, acknowledgments, or affiliations are
   present.
3. Verify the anonymous URL while signed out and ensure no access request is sent
   to an author email address.
4. Compare the uploaded ZIP hash with
   `eda-review-anonymous.zip.sha256`.
5. Keep the review commit unchanged. For a necessary correction, create
   `review-v2`, document the change, and update both reviewer routes.
6. Do not claim that code is public when it is available only through a private
   source repository or an anonymous review link.

## Suggested Response-letter Wording

Replace the placeholders with the tested anonymous URL and snapshot identifiers.
Mention only access routes that actually exist.

```tex
\noindent\textbf{Response:}
Thank you for this suggestion. We have prepared a complete reproducibility
package containing the EDA implementation, attack-generation and evaluation
scripts, environment specifications, dataset and checkpoint preparation
instructions, random seeds, and detailed documentation. The identity-scrubbed
snapshot used for the revised experiments is available to the reviewers through
the read-only anonymous repository at \url{<ANONYMOUS_REVIEW_URL>}. The same
snapshot is provided as \texttt{<SUPPLEMENTARY_ARCHIVE_NAME>} and is identified
by commit \texttt{<REVIEW_COMMIT>} and SHA-256 \texttt{<ARCHIVE_SHA256>}.
Dataset images, third-party model weights, generated outputs, provider responses,
and API credentials are not redistributed. Upon acceptance, we will publish the
reviewed snapshot in a versioned repository and create a persistent archival
release.
```

## Suggested Manuscript Wording During Review

```tex
\section*{Code availability}
The complete code and reproducibility package for the revised experiments are
available to reviewers through a read-only anonymous repository and the
supplementary code archive. The package includes environment specifications,
data and checkpoint preparation instructions, attack-generation and evaluation
scripts, random seeds, and expected output formats. The reviewed snapshot will
be made publicly available in a versioned archival release upon acceptance.
```

## Public Release After Acceptance

After acceptance, publish the reviewed snapshot, tag the article version (for
example, `v1.0.0`), create a GitHub Release, and archive it through a service that
provides a persistent DOI. Update `CITATION.cff`, README, and the manuscript with
the public URL and DOI. Continue to distribute data and third-party model weights
through preparation instructions rather than bundling them.


# Security

## Reporting

Please open a [private security advisory](https://github.com/CoderSoham/NLP_FIR/security/advisories/new)
rather than a public issue, and allow a week for an acknowledgement.

## Handling call data

This application processes emergency-call audio, which routinely contains
names, phone numbers, addresses and medical details.

- Uploads and generated artefacts are deleted after `RETENTION_SECONDS`
  (one hour by default) by a sweeper that also runs at startup.
- Uploads are stored outside the static mount and served only through a route
  that accepts names this application generated.
- Reports and plots are keyed to the request that produced them, and both path
  segments are validated before the filesystem is touched.

## Do not commit call data

Never commit recordings, transcripts or anything derived from real calls.
`.gitignore` excludes `models/` and common audio extensions, but treat that as
a backstop rather than a guarantee — check `git status` before staging.

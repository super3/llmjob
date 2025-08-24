# Instructions for Claude

## Start of Conversation

Always read this CLAUDE.md file at the start of each conversation to ensure you follow project-specific rules and workflows.

## Testing Requirements

Always run tests before starting work and after completing tasks. A task is NOT complete unless all tests pass. Use `npm test` in the appropriate directory to verify code quality and functionality.

## Git Workflow Rules

- NEVER commit unless explicitly requested with words like "commit", "push", or "save"
- When user says "Push" - this means commit AND push
- When making edits, just edit and stop - don't commit
- After making changes, wait for user's next instruction
- If asked to commit, show what will be committed first (git status)
- Use descriptive commit messages that explain the "why" not just the "what"
- Never commit or push changes unless explicitly requested
- This maintains control over what gets pushed to the repository
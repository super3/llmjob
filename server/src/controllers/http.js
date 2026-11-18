// Shared response handling for the route handlers.

// Wrap an async route handler so any error it throws becomes a response.
//
// An error carrying a 4xx `status` is a deliberate rejection, so its message
// goes back to the caller with that status. Anything else is a server fault:
// it is logged under `msg` and the caller gets a 500 with `msg` alone, so no
// driver message or stack detail leaks.
function handle(fn, msg) {
  return async function handled(req, res) {
    try {
      return await fn.call(this, req, res);
    } catch (error) {
      if (error && error.status >= 400 && error.status < 500) {
        return res.status(error.status).json({ error: error.message });
      }
      console.error(msg + ':', error);
      return res.status(500).json({ error: msg });
    }
  };
}

// Send a service's `{ error, status }` result as an error response (400 when it
// names no status).
function replyError(res, result) {
  return res.status(result.status || 400).json({ error: result.error });
}

// Send a service result: its error if it has one, otherwise the result itself.
function reply(res, result) {
  if (result.error) return replyError(res, result);
  return res.json(result);
}

module.exports = { handle, replyError, reply };

This is a fork of https://github.com/StonyBrookNLP/appworld

The main purpose of this fork is to improve the MCP server implementation, so that it handles the authorization internally, rather than requiring the agent to use the supervisor API for email and password retrieval, and the individual app login APIs to generate a token.  All this happens within the MCP server, at least for the file_system and gmail apps.  There are also fixes to list tools and other minor changes.

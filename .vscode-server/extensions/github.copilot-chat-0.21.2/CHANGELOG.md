## 0.21 (2024-10-02)

GitHub Copilot updates from [September 2024](https://code.visualstudio.com/updates/v1_94):

### Switch language models in chat

Previously, we announced that you can [sign up for early access to OpenAI o1 models](https://github.com/o1-waitlist-signup). Once you have access, you will have a [**Copilot Chat model picker**](https://code.visualstudio.com/updates/v1_94#_github-copilot) control in Copilot Chat in VS Code to choose which model version to use for your chat conversations.

![Copilot model picker control in the Chat view enables switching to another language model.](https://code.visualstudio.com/assets/updates/1_94/copilot-model-picker.png)

### Use GPT-4o in Inline Chat

We've upgraded Copilot Inline Chat to GPT-4o, to give you faster, more accurate, and higher-quality code and explanations when you use Chat in the editor.

### Public code matching in chat

You can allow GitHub Copilot to return code that could match publicly available code on GitHub.com. When this functionality is enabled for your [organization subscription](https://docs.github.com/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/setting-policies-for-copilot-in-your-organization/managing-policies-for-copilot-in-your-organization#policies-for-suggestion-matching) or [personal subscription](https://docs.github.com/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/managing-copilot-policies-as-an-individual-subscriber#enabling-or-disabling-suggestions-matching-public-code), Copilot code completions already provided you with details about the matches that were detected. We now show you these matches for public code in Copilot Chat as well.

If this is enabled for your organization or subscription, you might see a message at the end of the response with a **View matches** link. If you select the link, an editor opens that shows you the details of the matching code references with more details.

![Chat code referencing example.](https://code.visualstudio.com/assets/updates/1_94/code-references.png)

Get more information about [code referencing in GitHub Copilot](https://github.blog/news-insights/product-news/code-referencing-now-generally-available-in-github-copilot-and-with-microsoft-azure-ai/) on the GitHub Blog.

### File suggestions in chat

In chat input fields you can now type `#<filename>` to get file names suggested and attached. This works in chat locations that support file attachments, such as panel-chat, quick-chat, inline- and notebook-chat.

<video src="https://code.visualstudio.com/assets/updates/1_94/chat-file-complete.mp4" title="File suggestions when typing #filename" autoplay loop controls muted></video>

### Improved file links in chat responses

We've improved rendering of any workspace file paths mentioned in Copilot responses. These paths are very common when asking [`@workspace`](https://code.visualstudio.com/docs/copilot/copilot-chat.md#workspace) questions.

The first thing you'll notice is that paths to workspace files now include a file icon so that the type of file can be easily distinguished (these file icons are based on your current [file icon theme](https://code.visualstudio.com/docs/getstarted/themes.md#file-icon-themes)):

![Paths to workspace files in the response now render using file icons](https://code.visualstudio.com/assets/updates/1_94/copilot-path-overview.png)

These paths are clickable links, so just click on them to open the corresponding file. You can even use drag and and drop to open the file in a new editor group or insert it into a text editor by holding <kbd>shift</kbd> before dropping:

<video src="https://code.visualstudio.com/assets/updates/1_94/copilot-path-dnd.mp4" title="Dragging and dropping a workspace file from copilot into the editor" autoplay loop controls muted></video>

By default these links only show the file name but you can hover over them to see the full file path:

![Hovering over a workspace path to see the full workspace path](https://code.visualstudio.com/assets/updates/1_94/copilot-path-hover.png)

You can also right click on one of these paths to open a context menu with additional commands, including copying a relative path to the resource or revealing it in your operating system's explorer:

![The right context menu for a workspace path](https://code.visualstudio.com/assets/updates/1_94/copilot-path-right-click.png)

We plan to continue improving workspace path rendering in the coming iterations, as well as making similar improvements to symbol names in responses.

### Drag and drop files to add chat context

You can now easily attach additional files as context for a chat prompt by dragging files or editor tabs from the workbench directly into chat. By holding `Shift`, you can drop a file into Inline Chat instead of opening it in the editor to add it as context.

<video src="https://code.visualstudio.com/assets/updates/1_94/copilot-attach-dnd.mp4" title="Dragging files and editors into chat" autoplay loop controls muted></video>

### File attachments included in history

There are multiple ways to attach a file or editor selection as relevant context to your chat request, for example by using the 📎 button or typing `#`. Previously, this context was added only for the current request but was not included in the history of follow-on requests. Now, these attachments are kept in history, so that you can keep asking about them without having to reattach this context.

![persistent file attachments](https://code.visualstudio.com/assets/updates/1_94/file-attachment.png)

### Inline Chat and completions in Python native REPL

The native REPL editor used by the Python extension now supports Copilot Inline Chat and code completions directly in the input box.

<video src="https://code.visualstudio.com/assets/updates/1_94/copilot-in-REPL.mp4" title="Title" autoplay loop controls muted></video>

### Accept and run in notebook

When you use Copilot to generate code in a notebook, you can now accept the response and run it directly from the Inline Chat toolbar.

<video src="https://code.visualstudio.com/assets/updates/1_94/notebook-accept-run.mp4" title="Title" autoplay loop controls muted></video>

### Attach variable in notebook chat requests

When using Copilot in a notebook, you can now attach variables from Jupyter kernel in your requests, via either `#kernelVariable` completions, or by using the **Attach Context** (`kb(workbench.action.chat.attachContext`) action from the Inline Chat control. Adding variables gives you more precise control over the context of your requests in Jupyter notebooks.

<video src="https://code.visualstudio.com/assets/updates/1_94/notebook-kernel-variable.mp4" title="Title" autoplay loop controls muted></video>

### Refreshed welcome view and chat input

We've refreshed the chat panel with a clean new welcome view, and we've updated the layout of the chat input. We've added a `@` button to make it easier to find chat participants that are built-in or from chat extensions that you've installed. You can also still find these by typing `/` or `@` as you could before.

![welcome view](https://code.visualstudio.com/assets/updates/1_94/chat-welcome.png)

### Semantic search results

**Setting**: `github.copilot.chat.search.semanticTextResults`

The Search view enables you to perform an exact search across your files. We have now added a semantic search functionality to the Search view that uses Copilot to give search results that are semantically relevant.

Notice in the screenshot that the text results only contain exact matches for "diff view", whereas the GitHub Copilot results also have relevant matches for "merge editor".

<video controls src="https://code.visualstudio.com/assets/updates/1_94/semantic-search-in-search-view.mp4" title="Semantic Search in Search View"></video>

This functionality is still in preview and by default, the setting is not enabled. Try it out and let us know what you think!

### Fix test failure (Preview)

**Setting**: `github.copilot.chat.fixTestFailure.enabled`

We've added specialized logic to help you to diagnose the reasons why unit tests fail. This logic is triggered in some scenarios with `/fix`, and you can also invoke it through the `/fixTestFailure` slash command. The command is enabled in chat by default but can be disabled via the setting `github.copilot.chat.fixTestFailure.enabled`.

### Automated test setup (Experimental)

**Setting**: `github.copilot.chat.experimental.setupTests.enabled`

We added an experimental `/setupTests` slash command that can recommend a testing framework for your workspace, provide steps to setup and configure it, and recommend a VS Code extension to provide [testing integration in VS Code](https://code.visualstudio.com/docs/editor/testing). This can save you time and effort to get started with testing for your code.

When you use the `/tests` command to generate tests for your code, it can recommend `/setupTests` and testing extensions if looks like such an integration has not been set up yet in your workspace.

### Start debugging from Chat (Experimental)

**Setting**: `github.copilot.chat.experimental.startDebugging.enabled`

In this milestone, we made improvements to the experimental `/startDebugging` slash command. This command enables you to easily find or create a launch configuration and start debugging your application seamlessly. When you use `@vscode` in Copilot Chat, `/startDebugging` is now available by default.

![A user types /startDebugging flask app port 3000 in the panel chat and is provided with the launch configuration](https://code.visualstudio.com/assets/updates/1_94/start-debugging.png)

### Chat in Command Center (Experimental)

We are experimenting with a command center entry for chat. It provides quick access to all chat relevant commands, like starting chat or attaching context. You can enable this via `chat.commandCenter.enabled` but note that the command center itself needs to be enabled as well.

![Chat Command Center](https://code.visualstudio.com/assets/updates/1_94/chat-command-center.png)

### Improved temporal context (Experimental)

With the `github.copilot.chat.experimental.temporalContext.enabled` setting, you can instruct Inline Chat to consider files that you have opened or edited recently. We have improved this feature and invite everyone to give it a go.

## Previous release: https://code.visualstudio.com/updates

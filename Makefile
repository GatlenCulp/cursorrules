.PHONY: install install-force clean help

# Default target
.DEFAULT_GOAL := help

# Install .cursor to ~/.cursor without overwriting existing files
install:
	@echo "Installing .cursor to ~/.cursor (no-clobber mode)..."
	@mkdir -p ~/.cursor
	@cp -rn .cursor/* ~/.cursor/
	@echo "✓ Installation complete. Existing files were preserved."
	@echo "Note: Files that already existed were NOT overwritten."

# Install .cursor to ~/.cursor, prompting for overwrites
install-interactive:
	@echo "Installing .cursor to ~/.cursor (interactive mode)..."
	@mkdir -p ~/.cursor
	@cp -ri .cursor/* ~/.cursor/
	@echo "✓ Installation complete."

# Force install, overwriting all files
install-force:
	@echo "Installing .cursor to ~/.cursor (force mode)..."
	@mkdir -p ~/.cursor
	@cp -rf .cursor/* ~/.cursor/
	@echo "✓ Installation complete. All files were overwritten."

# Remove installed files from ~/.cursor
clean:
	@echo "Removing installed files from ~/.cursor..."
	@echo "Warning: This will remove files that match the structure in .cursor/"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		for file in .cursor/*; do \
			basename=$$(basename $$file); \
			if [ -e ~/.cursor/$$basename ]; then \
				rm -rf ~/.cursor/$$basename; \
				echo "Removed ~/.cursor/$$basename"; \
			fi; \
		done; \
		echo "✓ Cleanup complete."; \
	else \
		echo "Cancelled."; \
	fi

# Show help
help:
	@echo "Available targets:"
	@echo "  make install              - Install .cursor to ~/.cursor (no overwrite)"
	@echo "  make install-interactive  - Install .cursor to ~/.cursor (prompt on conflicts)"
	@echo "  make install-force        - Install .cursor to ~/.cursor (overwrite all)"
	@echo "  make clean                - Remove installed files from ~/.cursor"
	@echo "  make help                 - Show this help message"


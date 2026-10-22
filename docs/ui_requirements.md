# Story Engine UI Requirements - Phase 1

## Overview
Local desktop application using PySide6/Qt. No browser. Responsive with background generation.

## Screen Flow

### 1. Initial Screen - Project List
- Grid/list of all projects as clickable panels/cards
- Each panel shows: Project Name, Description (truncated), Created date, Character count
- Button "New Project" top-right
- Double-click or click opens Project View
- Right-click context menu for Edit/Delete project

### 2. Create/Edit Project Modal
- Simple modal dialog
- Fields: Name (required), Description (optional)
- Buttons: Cancel / Save
- On save, creates new project or updates existing

### 3. Project View
- Breadcrumb navigation: Home > [Project Name]
- Header with project name, description, actions (Edit Project, Delete Project, New Character)
- Characters section: Grid of character thumbnails with name underneath
- Each character card shows reference image thumbnail, name, version count
- Click character opens Character View
- Delete button per character card

### 4. Character View
- Breadcrumb: Home > [Project] > [Character Name]
- Left panel: Reference image (large), name, prompt, seed, model, created date
- Right panel: Versions gallery - all generated variants as thumbnails
- Can select different version to set as default/reference
- Buttons: Generate New Versions, Edit Character Details, Delete Character
- Edit allows rename, adjust attributes

### 5. Character Creation Builder
- Fields: Name, generation model, and one editable character prompt
- Optional image reference: upload an existing image and generate a prompt from its visual description
- Image-based generation keeps the reference pose and shows the source and generated preview side by side
- Edit the prompt and generate another preview; save only after reviewing the current preview
- Without a reference image, generate and review a text-to-image preview before saving
- After saving, goes to Character View

## Data Model Extensions

### Character Attributes
Existing character attribute data remains available for compatibility, but the creation builder now uses the prompt as the character description.

### Prompt Generation
The optional image-to-prompt action uses an image-captioning model to produce an editable description that includes pose and framing guidance.

## Navigation
Stack-based navigation with back button. Breadcrumb reflects history.

## File Structure
ui/
  main.py - App entry point
  screens/
    project_list_screen.py
    project_view_screen.py
    character_view_screen.py
    character_builder_screen.py
  components/
    project_card.py
    character_card.py
    breadcrumb.py
  dialogs/
    project_dialog.py
    confirm_dialog.py

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
- Step-by-step form with attributes:
  - Basic: Name, Gender/Sex, Age range
  - Appearance: Body type, Height, Build
  - Hair: Hair type, Hair color, Length, Style
  - Face: Skin tone, Eye color, Facial features
  - Clothing: Style, Color, Era
  - Mood/Expression: Happy, Serious, etc.
- LLM prompt generation: Use TEXT_GENERATION_MODELS (Phi-3/Gemma) to convert attributes into detailed prompt
- Preview: Generate 1 preview image before committing
- Options: Number of variants to generate (1-10)
- After creation, goes to Character View

## Data Model Extensions

### Character Attributes
New table `character_attributes`:
- id, project, character_name, attribute_json (JSON blob), created_at, updated_at
Attributes stored as JSON for flexibility:
{
  "gender": "female",
  "age_range": "20-30",
  "body_type": "athletic",
  "hair_type": "straight",
  "hair_color": "brown",
  "skin_tone": "medium",
  "eye_color": "blue",
  ...
}

### Prompt Generation
Use LLM model to convert attributes JSON -> detailed prompt:
"Create a photorealistic portrait of a [attributes] character..."

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

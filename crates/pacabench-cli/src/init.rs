//! Project initialization command.

use anyhow::{anyhow, Result};
use handlebars::Handlebars;
use rust_embed::Embed;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use toml_edit::{Array, DocumentMut, Item, Table, Value};

#[derive(Embed)]
#[folder = "templates/init/"]
struct InitTemplates;

/// Read project name from pyproject.toml if it exists.
pub fn read_pyproject_name() -> Option<String> {
    let path = PathBuf::from("pyproject.toml");
    let content = fs::read_to_string(&path).ok()?;
    let doc: DocumentMut = content.parse().ok()?;
    doc.get("project")?
        .get("name")?
        .as_str()
        .map(|s| s.to_string())
}

pub fn execute(name: &str, force: bool, update_pyproject: bool) -> Result<()> {
    let mut created = Vec::new();
    let mut skipped = Vec::new();
    let mut handlebars = Handlebars::new();
    handlebars.set_strict_mode(true);
    let context = json!({ "name": name });

    for entry in InitTemplates::iter() {
        let template_path = entry.as_ref();
        let embedded = InitTemplates::get(template_path)
            .ok_or_else(|| anyhow!("missing template {template_path}"))?;
        let output_path = target_path(template_path);

        if output_path.as_os_str() == ".gitignore" {
            handle_gitignore(
                &output_path,
                embedded.data.as_ref(),
                &mut created,
                &mut skipped,
            )?;
            continue;
        }

        if output_path.exists() && !force {
            skipped.push(output_path.display().to_string());
            continue;
        }

        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let bytes = if template_path.ends_with(".hbs") {
            let template_str = std::str::from_utf8(&embedded.data)?;
            handlebars
                .render_template(template_str, &context)?
                .into_bytes()
        } else {
            embedded.data.to_vec()
        };

        fs::write(&output_path, bytes)?;
        created.push(output_path.display().to_string());
    }

    // Update pyproject.toml if requested
    if update_pyproject {
        let pyproject_path = PathBuf::from("pyproject.toml");
        if pyproject_path.exists() {
            update_pyproject_toml(&pyproject_path)?;
            created.push("pyproject.toml (updated)".into());
        } else {
            println!(
                "Note: No pyproject.toml found. Add pacabench manually:\n  \
                 uv add --dev pacabench\n  or\n  \
                 pip install pacabench"
            );
        }
    }

    print_summary(&created, &skipped);
    println!("\nNext steps:");
    println!("  1. Set your API key: export OPENAI_API_KEY=sk-...");
    println!("  2. Install dependencies");
    println!("  3. Run: uv run pacabench run --limit 3");
    println!("\nSee https://github.com/fastpaca/pacabench for docs.");

    Ok(())
}

fn target_path(template_path: &str) -> PathBuf {
    match template_path {
        "gitignore.txt" => PathBuf::from(".gitignore"),
        _ if template_path.ends_with(".hbs") => {
            PathBuf::from(template_path.trim_end_matches(".hbs"))
        }
        _ => PathBuf::from(template_path),
    }
}

fn handle_gitignore(
    path: &Path,
    template_data: &[u8],
    created: &mut Vec<String>,
    skipped: &mut Vec<String>,
) -> Result<()> {
    let entry = std::str::from_utf8(template_data)?.trim();
    if path.exists() {
        let mut existing = fs::read_to_string(path)?;
        if existing.contains(entry) {
            skipped.push(path.display().to_string());
        } else {
            if !existing.ends_with('\n') {
                existing.push('\n');
            }
            existing.push_str(entry);
            existing.push('\n');
            fs::write(path, existing)?;
            created.push(format!("{} (updated)", path.display()));
        }
    } else {
        fs::write(path, format!("{entry}\n"))?;
        created.push(path.display().to_string());
    }
    Ok(())
}

fn print_summary(created: &[String], skipped: &[String]) {
    if !created.is_empty() {
        println!("Created pacabench project:");
        for path in created {
            println!("  {path}");
        }
    }
    if !skipped.is_empty() {
        println!("\nSkipped existing files:");
        for path in skipped {
            println!("  {path}");
        }
    }
}

fn update_pyproject_toml(path: &PathBuf) -> Result<()> {
    let content = fs::read_to_string(path)?;
    let mut doc: DocumentMut = content.parse()?;

    if let Some(array) = dependency_groups_dev(&mut doc) {
        let updated = ensure_dependency(array, "pacabench");
        if updated {
            fs::write(path, doc.to_string())?;
        }
        return Ok(());
    }

    if let Some(array) = optional_dependencies_dev(&mut doc) {
        let updated = ensure_dependency(array, "pacabench");
        if updated {
            fs::write(path, doc.to_string())?;
        }
        return Ok(());
    }

    // Neither dependency section exists - create dependency-groups.dev
    let mut dev_array = Array::new();
    dev_array.push("pacabench");
    let mut dep_groups = Table::new();
    dep_groups.insert("dev", Item::Value(Value::Array(dev_array)));
    doc.insert("dependency-groups", Item::Table(dep_groups));

    fs::write(path, doc.to_string())?;
    Ok(())
}

fn dependency_groups_dev(doc: &mut DocumentMut) -> Option<&mut Array> {
    doc.get_mut("dependency-groups")
        .and_then(Item::as_table_mut)
        .and_then(|table| table.get_mut("dev"))
        .and_then(Item::as_array_mut)
}

fn optional_dependencies_dev(doc: &mut DocumentMut) -> Option<&mut Array> {
    doc.get_mut("project")
        .and_then(Item::as_table_mut)
        .and_then(|table| table.get_mut("optional-dependencies"))
        .and_then(Item::as_table_mut)
        .and_then(|table| table.get_mut("dev"))
        .and_then(Item::as_array_mut)
}

fn ensure_dependency(array: &mut Array, dep: &str) -> bool {
    let exists = array
        .iter()
        .any(|value| value.as_str().map(|s| s == dep).unwrap_or(false));
    if exists {
        false
    } else {
        array.push(dep);
        true
    }
}

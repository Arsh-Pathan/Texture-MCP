# Integrating Texture-Craft with an AI Minecraft Plugin Generator

This guide explains how an **AI Minecraft Plugin Generator** can use Texture-Craft to automatically produce **textures, item models, CustomModelData overrides, and Spigot/Paper Java code** whenever it generates custom items for a plugin.

---

## 🏗️ How Custom Plugin Items Work in Minecraft

Minecraft (Java Edition 1.14+) allows server plugins to define custom items **without client-side mods** (e.g. Vanilla clients can join) using **`CustomModelData`**:

```
[ AI Plugin Generator ]
        │
        ├── 1. Generates Spigot/Paper Java Plugin Code
        │      (e.g., /givemagic command, item lore, custom abilities)
        │
        └── 2. Calls Texture-Craft API (/v1/resourcepack/item)
               │
               ▼
   [ Resource Pack Output ]
   ├── pack.mcmeta
   └── assets/
       ├── myplugin/
       │   ├── textures/item/frost_sword.png   <-- 16x16 RGBA texture
       │   └── models/item/frost_sword.json    <-- Custom item model
       └── minecraft/
           └── models/item/diamond_sword.json  <-- Override predicate (CustomModelData: 1001)
```

When a player holds a `DIAMOND_SWORD` with `CustomModelData: 1001`, the client automatically displays the custom `frost_sword` texture!

---

## 🚀 Step 1: Start Texture-Craft API

Run Texture-Craft as a background service on your system:

```powershell
Set-Location "C:\Users\ArshPathan\Projects\CSky-Developments\sourcecode\Texture-Craft"
.\.venv\Scripts\python.exe server.py
```

The server starts at `http://127.0.0.1:8000`.

---

## 🐍 Step 2: Python Client (for Python Plugin Generators)

If your AI plugin generator is written in Python, add this helper module:

```python
import requests
from typing import Dict, Any, Optional

TEXTURE_CRAFT_URL = "http://127.0.0.1:8000"

class TextureCraftClient:
    def __init__(self, base_url: str = TEXTURE_CRAFT_URL):
        self.base_url = base_url.rstrip("/")

    def is_ready(self) -> bool:
        try:
            res = requests.get(f"{self.base_url}/health", timeout=3)
            return res.status_code == 200 and res.json().get("status") == "ready"
        except Exception:
            return False

    def init_resource_pack(self, pack_dir: str, pack_name: str = "PluginPack") -> Dict[str, Any]:
        payload = {
            "pack_dir": pack_dir,
            "pack_name": pack_name,
            "description": f"Resource pack for {pack_name}",
            "pack_format": 34  # 34 = 1.21/1.21.1
        }
        res = requests.post(f"{self.base_url}/v1/resourcepack/init", json=payload)
        res.raise_for_status()
        return res.json()

    def create_custom_item(
        self,
        pack_dir: str,
        namespace: str,
        item_id: str,
        prompt: str,
        base_vanilla_item: str,
        custom_model_data: int,
        display_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generates texture, writes model JSON, and returns Java code snippet."""
        payload = {
            "pack_dir": pack_dir,
            "namespace": namespace,
            "item_id": item_id,
            "prompt": prompt,
            "base_vanilla_item": base_vanilla_item,
            "custom_model_data": custom_model_data,
            "display_name": display_name,
            "seed": seed,
        }
        res = requests.post(f"{self.base_url}/v1/resourcepack/item", json=payload)
        res.raise_for_status()
        return res.json()


# ---------------- Example Usage in Generator ----------------
if __name__ == "__main__":
    client = TextureCraftClient()
    pack_path = "C:/my_minecraft_server/plugins/MyMagicPlugin/resourcepack"

    # 1. Initialize pack
    client.init_resource_pack(pack_path, pack_name="MyMagicPluginPack")

    # 2. Generate item
    item_result = client.create_custom_item(
        pack_dir=pack_path,
        namespace="mymagic",
        item_id="lightning_wand",
        prompt="golden lightning wand with electric spark orb",
        base_vanilla_item="stick",
        custom_model_data=2001,
        display_name="Lightning Wand"
    )

    print("Created texture at:", item_result["texture_file"])
    print("Java Snippet to paste into plugin:\n")
    print(item_result["java_code_snippet"])
```

---

## 🟨 Step 3: TypeScript / Node.js Client (for Web/JS Generators)

If your AI plugin generator is written in Node.js, Next.js, or TypeScript:

```typescript
import axios from 'axios';

interface CreateItemOptions {
  packDir: string;
  namespace: string;
  itemId: string;
  prompt: string;
  baseVanillaItem: string;
  customModelData: number;
  displayName?: string;
  seed?: number;
}

export class TextureCraftClient {
  private baseUrl: string;

  constructor(baseUrl = 'http://127.0.0.1:8000') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async initPack(packDir: string, packName = 'PluginPack') {
    const res = await axios.post(`${this.baseUrl}/v1/resourcepack/init`, {
      pack_dir: packDir,
      pack_name: packName,
      description: `Resource pack for ${packName}`,
      pack_format: 34
    });
    return res.data;
  }

  async createCustomItem(opts: CreateItemOptions) {
    const res = await axios.post(`${this.baseUrl}/v1/resourcepack/item`, {
      pack_dir: opts.packDir,
      namespace: opts.namespace,
      item_id: opts.itemId,
      prompt: opts.prompt,
      base_vanilla_item: opts.baseVanillaItem,
      custom_model_data: opts.customModelData,
      display_name: opts.displayName,
      seed: opts.seed
    });
    return res.data;
  }
}
```

---

## ☕ Step 4: Spigot / Paper Java Code Output

Texture-Craft automatically generates matching Java code. Here is how your AI generator should assemble the generated items inside a Spigot/Paper `JavaPlugin`:

```java
package com.example.mymagic;

import org.bukkit.Material;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.ItemMeta;
import java.util.List;

public class CustomItems {

    public static ItemStack createLightningWand() {
        ItemStack item = new ItemStack(Material.STICK);
        ItemMeta meta = item.getItemMeta();
        if (meta != null) {
            meta.setDisplayName("§6§lLightning Wand");
            meta.setCustomModelData(2001); // Matches Texture-Craft override
            meta.setLore(List.of(
                "§7Casts a thunderous strike upon foes.",
                "§eRight-click to invoke lightning!"
            ));
            item.setItemMeta(meta);
        }
        return item;
    }
}
```

---

## 📦 Step 5: Distributing the Resource Pack

Your plugin can deliver the generated resource pack to players using either:
1. **Server Resource Pack**: Set `resource-pack=` in `server.properties` or call `player.setResourcePack("http://your-server.com/pack.zip")` in Paper.
2. **ItemsAdder / Oraxen**: If the user uses ItemsAdder or Oraxen, the generated 16×16 textures can be copied directly into their texture directories.

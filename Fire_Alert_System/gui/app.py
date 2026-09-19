import os
import cv2
import queue
import config
import traceback
import tkinter as tk
from src.db import Database
from PIL import Image, ImageTk
from src.video import VideoWorker
from src.alert import AlertManager
from src.tracker import FireDetector
from tkinter import filedialog, messagebox, ttk


class FireAlertApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(config.App_title)
        self.geometry("1150x650")
        self.minsize(950, 600)
        config.ensure_directories()
        self.database = Database(config.DB_path)
        self.detector = FireDetector()
        self.alert_manager = AlertManager(
            database=self.database,
            on_notify=self._queue_in_app_notification,)
        self.video_worker = VideoWorker(
            self.detector,
            self.alert_manager,)
        self._pending_notifications = queue.Queue()
        self._build_layout()
        self._refresh_history()
        self.after(15, self._poll_video_queue)
        self.after(300, self._poll_notifications)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # Layout
    # ==================================================================

    def _build_layout(self):
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.upload_button = ttk.Button(
            toolbar,
            text="Upload Video",
            command=self._on_upload_video,
        )
        self.upload_button.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_button = ttk.Button(
            toolbar,
            text="Stop",
            command=self._on_stop,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(
            value="Idle — upload a video to begin monitoring."
        )

        ttk.Label(
            toolbar,
            textvariable=self.status_var,
        ).pack(side=tk.LEFT, padx=16)

        body = ttk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
        )

        body.pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=8,
        )

        video_frame = ttk.Frame(body)
        history_frame = ttk.Frame(body)

        body.add(video_frame, weight=3)
        body.add(history_frame, weight=2)

        self.video_label = ttk.Label(
            video_frame,
            background="#1e1e1e",
        )

        self.video_label.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self._build_history_panel(history_frame)

    # ------------------------------------------------------------------
    # History panel
    # ------------------------------------------------------------------

    def _build_history_panel(self, parent: ttk.Frame):
        ttk.Label(
            parent,
            text="Alert History",
            font=("Segoe UI", 12, "bold"),
        ).pack(
            anchor=tk.W,
            pady=(0, 6),
        )

        columns = (
            "timestamp",
            "behavior",
            "confidence",
        )

        self.history_tree = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            height=18,
        )

        self.history_tree.heading(
            "timestamp",
            text="Date / Time",
        )

        self.history_tree.heading(
            "behavior",
            text="Behavior",
        )

        self.history_tree.heading(
            "confidence",
            text="Confidence",
        )

        self.history_tree.column(
            "timestamp",
            width=140,
        )

        self.history_tree.column(
            "behavior",
            width=80,
            anchor=tk.CENTER,
        )

        self.history_tree.column(
            "confidence",
            width=80,
            anchor=tk.CENTER,
        )

        self.history_tree.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.history_tree.bind(
            "<Double-1>",
            self._on_history_row_double_click,
        )

        self._row_to_screenshot = {}

        button_row = ttk.Frame(parent)
        button_row.pack(
            fill=tk.X,
            pady=6,
        )

        ttk.Button(
            button_row,
            text="Refresh",
            command=self._refresh_history,
        ).pack(side=tk.LEFT)

        ttk.Button(
            button_row,
            text="Open Screenshot",
            command=self._open_selected_screenshot,
        ).pack(
            side=tk.LEFT,
            padx=6,
        )

        ttk.Button(
            button_row,
            text="Clear History",
            command=self._on_clear_history,
        ).pack(side=tk.RIGHT)

    # ==================================================================
    # Upload / processing controls
    # ==================================================================

    def _on_upload_video(self):
        path = filedialog.askopenfilename(
            title="Select an exam/security camera video",
            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        self.video_worker.start(path)

        self.status_var.set(
            f"Monitoring: {os.path.basename(path)}"
        )

        self.stop_button.configure(
            state=tk.NORMAL
        )

    def _on_stop(self):
        self.video_worker.stop()

        self.status_var.set(
            "Stopped."
        )

        self.stop_button.configure(
            state=tk.DISABLED
        )

    # ==================================================================
    # Video queue polling
    # ==================================================================

    def _poll_video_queue(self):
        try:
            while True:
                message = self.video_worker.frame_queue.get_nowait()
                if message.finished:
                    self.status_var.set(
                        "Video finished."
                    )

                    self.stop_button.configure(
                        state=tk.DISABLED
                    )

                    self._refresh_history()

                    break

                if message.frame is not None:
                    self._display_frame(
                        message.frame
                    )

        except queue.Empty:
            pass

        except Exception:
            traceback.print_exc()

        finally:
            self.after(
                15,
                self._poll_video_queue,
            )

    def _display_frame(self, frame):
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb_frame
        )

        image.thumbnail(
            (
                config.Video_width,
                config.Video_height,
            )
        )

        photo = ImageTk.PhotoImage(
            image=image
        )

        self.video_label.configure(
            image=photo
        )
        self.video_label.image = photo

    # ==================================================================
    # Notifications
    # ==================================================================

    def _queue_in_app_notification(
        self,
        title: str,
        message: str,
        screenshot_path: str,
    ):

        self._pending_notifications.put(
            (
                title,
                message,
                screenshot_path,
            )
        )

    def _poll_notifications(self) -> None:
        """
        Runs in the Tkinter/main thread.

        Retrieves notifications produced by AlertManager.
        """

        try:
            while True:
                title, message, screenshot_path = (
                    self._pending_notifications.get_nowait()
                )

                self._refresh_history()

                messagebox.showwarning(
                    title,
                    message,
                )

        except queue.Empty:
            pass

        except Exception:
            traceback.print_exc()

        finally:
            self.after(
                300,
                self._poll_notifications,
            )

    # ==================================================================
    # Alert history / screenshot viewer
    # ==================================================================

    def _refresh_history(self):
        """
        Reload all alert records from the database and rebuild the
        Treeview.
        """

        try:
            self.history_tree.delete(
                *self.history_tree.get_children()
            )

            self._row_to_screenshot.clear()
            records = self.database.fetch_all_alerts()

            for record in records:

                item_id = self.history_tree.insert(
                    "",
                    tk.END,
                    values=(
                        record.timestamp,
                        record.behavior,
                        f"{record.confidence:.0%}",
                    ),
                )

                self._row_to_screenshot[item_id] = (
                    record.screenshot_path
                )

        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Double-click history row
    # ------------------------------------------------------------------

    def _on_history_row_double_click(self, _event):
        self._open_selected_screenshot()

    # ------------------------------------------------------------------
    # Open screenshot
    # ------------------------------------------------------------------

    def _open_selected_screenshot(self):
        selection = self.history_tree.selection()

        if not selection:
            messagebox.showinfo(
                "Screenshot Viewer",
                "Select an alert row first.",
            )
            return

        path = self._row_to_screenshot.get(
            selection[0]
        )

        if not path:
            messagebox.showerror(
                "Screenshot Viewer",
                "No screenshot path is associated with this alert.",
            )
            return

        if not os.path.exists(path):
            messagebox.showerror(
                "Screenshot Viewer",
                "Screenshot file not found.",
            )
            return

        self._show_screenshot_window(path)

    # ------------------------------------------------------------------
    # Screenshot viewer window
    # ------------------------------------------------------------------

    def _show_screenshot_window(self, path: str):
        try:
            viewer = tk.Toplevel(self)

            viewer.title(
                os.path.basename(path)
            )

            image = Image.open(path)

            image.thumbnail(
                (
                    900,
                    700,
                )
            )

            photo = ImageTk.PhotoImage(
                image
            )

            label = ttk.Label(
                viewer,
                image=photo,
            )

            # Keep a reference to prevent garbage collection.
            label.image = photo

            label.pack(
                padx=10,
                pady=10,
            )

        except Exception as exc:
            messagebox.showerror(
                "Screenshot Viewer",
                f"Could not open screenshot:\n\n{exc}",
            )

    # ==================================================================
    # Clear history
    # ==================================================================

    def _on_clear_history(self):
        confirmed = messagebox.askyesno(
            "Clear History",
            "Delete all saved alert records?",
        )

        if not confirmed:
            return

        try:
            self.database.clear_all_alerts()
            self._refresh_history()

        except Exception:
            traceback.print_exc()

            messagebox.showerror(
                "Clear History",
                "Could not clear alert history.",
            )

    # ==================================================================
    # Shutdown
    # ==================================================================

    def _on_close(self):
        """
        Called when the user closes the main window.
        """

        try:
            self.video_worker.stop()

        except Exception:
            traceback.print_exc()

        finally:
            self.destroy()

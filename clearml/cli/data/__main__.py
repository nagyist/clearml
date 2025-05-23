import json
import os
import shutil
from argparse import ArgumentParser, HelpFormatter
from functools import partial
from typing import Sequence, Optional, Dict, Any

from pathlib2 import Path

import clearml.backend_api.session
from clearml.datasets import Dataset
from clearml.version import __version__

clearml.backend_api.session.Session.add_client("clearml-data", __version__)


def check_null_id(args: Any) -> None:
    if not getattr(args, "id", None):
        raise ValueError("Dataset ID not specified, add --id <dataset_id>")


def print_args(args: Any, exclude: Sequence[str] = ("command", "func", "verbose")) -> ():
    if not getattr(args, "verbose", None):
        return
    for arg in args.__dict__:
        if arg in exclude or args.__dict__.get(arg) is None:
            continue
        print("{}={}".format(arg, args.__dict__[arg]))


def restore_state(args: Any) -> Any:
    session_state_file = os.path.expanduser("~/.clearml_data.json")
    # noinspection PyBroadException
    try:
        with open(session_state_file, "rt") as f:
            state = json.load(f)
    except Exception:
        state = {}

    args.id = getattr(args, "id", None) or state.get("id")

    state = {str(k): str(v) if v is not None else None for k, v in args.__dict__.items() if not str(k).startswith("_")}
    # noinspection PyBroadException
    try:
        with open(session_state_file, "wt") as f:
            json.dump(state, f, sort_keys=True)
    except Exception:
        pass

    return args


def clear_state(state: Optional[Dict] = None) -> None:
    session_state_file = os.path.expanduser("~/.clearml_data.json")
    # noinspection PyBroadException
    try:
        with open(session_state_file, "wt") as f:
            json.dump(state or dict(), f, sort_keys=True)
    except Exception:
        pass


def cli() -> int:
    title = "clearml-data - Dataset Management & Versioning CLI"
    print(title)
    parser = ArgumentParser(  # noqa
        description=title,
        prog="clearml-data",
        formatter_class=partial(HelpFormatter, indent_increment=0, max_help_position=10),
    )
    subparsers = parser.add_subparsers(help="Dataset actions", dest="command")

    create = subparsers.add_parser("create", help="Create a new dataset")
    create.add_argument(
        "--parents",
        type=str,
        nargs="*",
        help="[Optional] Specify dataset parents IDs (i.e. merge all parents). "
        "Example: a17b4fID1 f0ee5ID2 a17b4f09eID3",
    )
    create.add_argument("--project", type=str, required=False, default=None, help="Dataset project name")
    create.add_argument("--name", type=str, required=True, default=None, help="Dataset name")
    create.add_argument("--version", type=str, required=False, default=None, help="Dataset version")
    create.add_argument(
        "--output-uri",
        type=str,
        required=False,
        default=None,
        help="Output URI for files in this dataset (deprecated, use '--storage' instead)",
    )
    create.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Remote storage to use for the dataset files (default: files_server). "
        "Examples: 's3://bucket/data', 'gs://bucket/data', 'azure://bucket/data', "
        "'/mnt/shared/folder/data'",
    )
    create.add_argument("--tags", type=str, nargs="*", help="Dataset user Tags")
    create.set_defaults(func=ds_create)

    add = subparsers.add_parser("add", help="Add files or links to the dataset")
    add.add_argument(
        "--id",
        type=str,
        required=False,
        help="Previously created dataset id. Default: previously created/accessed dataset",
    )
    add.add_argument(
        "--dataset-folder",
        type=str,
        default=None,
        help="Dataset base folder to add the files to (default: Dataset root)",
    )
    add.add_argument("--files", type=str, nargs="*", help="Files / folders to add.")
    add.add_argument(
        "--wildcard",
        type=str,
        nargs="*",
        help="Add specific set of files, denoted by these wildcards. Multiple wildcards can be passed",
    )
    add.add_argument(
        "--links",
        type=str,
        nargs="*",
        help=(
            "Links to files / folders to add. Supports s3, gs, azure links. "
            "Example: s3://bucket/data azure://bucket/folder"
        ),
    )
    add.add_argument(
        "--non-recursive",
        action="store_true",
        default=False,
        help="Disable recursive scan of files",
    )
    add.add_argument("--verbose", action="store_true", default=False, help="Verbose reporting")
    add.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of threads to add the files with. Defaults to the number of logical cores",
    )
    add.set_defaults(func=ds_add)

    set_description = subparsers.add_parser("set-description", help="Set description to the dataset")
    set_description.add_argument(
        "--description",
        type=str,
        required=True,
        help="Description of the dataset",
    )
    set_description.add_argument(
        "--id",
        type=str,
        required=False,
        help="Previously created dataset id. Default: previously created/accessed dataset",
    )
    set_description.set_defaults(func=ds_set_description)

    sync = subparsers.add_parser("sync", help="Sync a local folder with the dataset")
    sync.add_argument(
        "--id",
        type=str,
        required=False,
        help="Previously created dataset id. Default: previously created/accessed dataset",
    )
    sync.add_argument(
        "--dataset-folder",
        type=str,
        default=None,
        help="Dataset base folder to add the files to (default: Dataset root)",
    )
    sync.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Local folder to sync (support for wildcard selection). Example: ~/data/*.jpg",
    )
    sync.add_argument(
        "--parents",
        type=str,
        nargs="*",
        help="[Optional] Specify dataset parents IDs (i.e. merge all parents). "
        "Example: a17b4fID1 f0ee5ID2 a17b4f09eID3",
    )
    sync.add_argument(
        "--project",
        type=str,
        required=False,
        default=None,
        help="[Optional] Dataset project name",
    )
    sync.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="[Optional] Dataset project name",
    )
    sync.add_argument(
        "--version",
        type=str,
        required=False,
        default=None,
        help="[Optional] Dataset version",
    )
    sync.add_argument(
        "--output-uri",
        type=str,
        required=False,
        default=None,
        help="[Optional] Output URI for artifacts/debug samples. Useable when creating the dataset (deprecated, use '--storage' instead)",
    )
    sync.add_argument("--tags", type=str, nargs="*", help="[Optional] Dataset user Tags")
    sync.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Remote storage to use for the dataset files (default: files_server). "
        "Examples: 's3://bucket/data', 'gs://bucket/data', 'azure://bucket/data', "
        "'/mnt/shared/folder/data'",
    )
    sync.add_argument(
        "--skip-close",
        action="store_true",
        default=False,
        help="Do not auto close dataset after syncing folders",
    )
    sync.add_argument(
        "--chunk-size",
        default=512,
        type=int,
        help="Set dataset artifact chunk size in MB. Default 512mb, (pass -1 for a single chunk). "
        "Example: 512, dataset will be split and uploaded in 512mb chunks.",
    )
    sync.add_argument("--verbose", action="store_true", default=False, help="Verbose reporting")
    sync.set_defaults(func=ds_sync)

    remove = subparsers.add_parser("remove", help="Remove files/links from the dataset")
    remove.add_argument(
        "--id",
        type=str,
        required=False,
        help="Previously created dataset id. Default: previously created/accessed dataset",
    )
    remove.add_argument(
        "--files",
        type=str,
        required=False,
        nargs="*",
        help="Files / folders to remove (support for wildcard selection). "
        "Notice: File path is the dataset path not the local path. "
        "Example: data/*.jpg data/jsons/",
    )
    remove.add_argument(
        "--non-recursive",
        action="store_true",
        default=False,
        help="Disable recursive scan of files",
    )
    remove.add_argument("--verbose", action="store_true", default=False, help="Verbose reporting")
    remove.set_defaults(func=ds_remove)

    upload = subparsers.add_parser("upload", help="Upload the local dataset changes to the server")
    upload.add_argument(
        "--id",
        type=str,
        required=False,
        help="Previously created dataset id. Default: previously created/accessed dataset",
    )
    upload.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Remote storage to use for the dataset files (default: files_server). "
        "Examples: 's3://bucket/data', 'gs://bucket/data', 'azure://bucket/data', "
        "'/mnt/shared/folder/data'",
    )
    upload.add_argument(
        "--chunk-size",
        default=512,
        type=int,
        help="Set dataset artifact chunk size in MB. Default 512, (pass -1 for a single chunk). "
        "Example: 512, dataset will be split and uploaded in 512mb chunks.",
    )
    upload.add_argument("--verbose", default=False, action="store_true", help="Verbose reporting")
    upload.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of threads to upload the files with. Defaults to 1 if uploading to a cloud provider ('s3', 'azure', 'gs') OR to the number of logical cores otherwise",
    )
    upload.set_defaults(func=ds_upload)

    finalize = subparsers.add_parser("close", help="Finalize and close the dataset (implies auto upload)")
    finalize.add_argument(
        "--id",
        type=str,
        required=False,
        help="Previously created dataset id. Default: previously created/accessed dataset",
    )
    finalize.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Remote storage to use for the dataset files (default: files_server). "
        "Examples: 's3://bucket/data', 'gs://bucket/data', 'azure://bucket/data', "
        "'/mnt/shared/folder/data'",
    )
    finalize.add_argument(
        "--disable-upload",
        action="store_true",
        default=False,
        help="Disable automatic upload when closing the dataset",
    )
    finalize.add_argument(
        "--chunk-size",
        default=512,
        type=int,
        help="Set dataset artifact chunk size in MB. Default 512, (pass -1 for a single chunk). "
        "Example: 512, dataset will be split and uploaded in 512mb chunks.",
    )
    finalize.add_argument("--verbose", action="store_true", default=False, help="Verbose reporting")
    finalize.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of threads to upload the files with. Defaults to 1 if uploading to a cloud provider ('s3', 'azure', 'gs') OR to the number of logical cores otherwise",
    )
    finalize.set_defaults(func=ds_close)

    publish = subparsers.add_parser("publish", help="Publish dataset task")
    publish.add_argument("--id", type=str, required=True, help="The dataset task id to be published.")
    publish.set_defaults(func=ds_publish)

    delete = subparsers.add_parser("delete", help="Delete a dataset")
    delete.add_argument(
        "--id",
        type=str,
        required=False,
        help="Previously created dataset id. Default: previously created/accessed dataset",
    )
    delete.add_argument(
        "--project",
        type=str,
        required=False,
        help="The project the dataset(s) to be deleted belong(s) to",
    )
    delete.add_argument(
        "--name",
        type=str,
        required=False,
        help="The name of the dataset(s) to be deleted",
    )
    delete.add_argument(
        "--version",
        type=str,
        required=False,
        help="The version of the dataset(s) to be deleted",
    )
    delete.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force dataset deletion even if other dataset versions depend on it. Must also be used if entire-dataset flag is used",
    )
    delete.add_argument(
        "--entire-dataset",
        action="store_true",
        default=False,
        help="Delete all found datasets",
    )
    delete.set_defaults(func=ds_delete)

    rename = subparsers.add_parser("rename", help="Rename a dataset")
    rename.add_argument("--new-name", type=str, required=True, help="The new name of the dataset(s)")
    rename.add_argument(
        "--project",
        type=str,
        required=True,
        help="The project the dataset(s) to be renamed belong(s) to",
    )
    rename.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the dataset(s) to be renamed",
    )
    rename.set_defaults(func=ds_rename)

    move = subparsers.add_parser("move", help="Move a dataset to another project")
    move.add_argument(
        "--new-project",
        type=str,
        required=True,
        help="The new project of the dataset(s)",
    )
    move.add_argument(
        "--project",
        type=str,
        required=True,
        help="The project the dataset(s) to be moved belong(s) to",
    )
    move.add_argument("--name", type=str, required=True, help="The name of the dataset(s) to be moved")
    move.set_defaults(func=ds_move)

    compare = subparsers.add_parser("compare", help="Compare two datasets (target vs source)")
    compare.add_argument("--source", type=str, required=True, help="Source dataset id (used as baseline)")
    compare.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target dataset id (compare against the source baseline dataset)",
    )
    compare.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose report all file changes (instead of summary)",
    )
    compare.set_defaults(func=ds_compare)

    squash = subparsers.add_parser(
        "squash",
        help="Squash multiple datasets into a single dataset version (merge down)",
    )
    squash.add_argument("--name", type=str, required=True, help="Create squashed dataset name")
    squash.add_argument(
        "--ids",
        type=str,
        required=True,
        nargs="*",
        help="Source dataset IDs to squash (merge down)",
    )
    squash.add_argument("--storage", type=str, default=None, help="See `upload storage`")
    squash.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose report all file changes (instead of summary)",
    )
    squash.set_defaults(func=ds_squash)

    search = subparsers.add_parser("search", help="Search datasets in the system (sorted by creation time)")
    search.add_argument("--ids", type=str, nargs="*", help="Specify list of dataset IDs")
    search.add_argument("--project", type=str, help="Specify datasets project name")
    search.add_argument("--name", type=str, help="Specify datasets partial name matching")
    search.add_argument("--tags", type=str, nargs="*", help="Specify list of dataset user tags")
    search.add_argument(
        "--not-only-completed",
        action="store_true",
        default=False,
        help="If set, return datasets that are still in progress as well",
    )
    search.add_argument(
        "--non-recursive-project-search",
        action="store_true",
        default=False,
        help="Don't search inside subprojects",
    )
    search.set_defaults(func=ds_search)

    verify = subparsers.add_parser("verify", help="Verify local dataset content")
    verify.add_argument(
        "--id",
        type=str,
        required=False,
        help="Specify dataset id. Default: previously created/accessed dataset",
    )
    verify.add_argument(
        "--folder",
        type=str,
        help="Specify dataset local copy (if not provided the local cache folder will be verified)",
    )
    verify.add_argument(
        "--filesize",
        action="store_true",
        default=False,
        help="If True, only verify file size and skip hash checks (default: false)",
    )
    verify.add_argument("--verbose", action="store_true", default=False, help="Verbose reporting")
    verify.set_defaults(func=ds_verify)

    ls = subparsers.add_parser("list", help="List dataset content")
    ls.add_argument(
        "--id",
        type=str,
        required=False,
        help="Specify dataset id (or use project/name instead). Default: previously accessed dataset.",
    )
    ls.add_argument("--project", type=str, help="Specify dataset project name")
    ls.add_argument("--name", type=str, help="Specify dataset name")
    ls.add_argument("--version", type=str, help="Specify dataset version", default=None)
    ls.add_argument(
        "--filter",
        type=str,
        nargs="*",
        help="Filter files based on folder / wildcard, multiple filters are supported. "
        "Example: folder/date_*.json folder/sub-folder",
    )
    ls.add_argument(
        "--modified",
        action="store_true",
        default=False,
        help="Only list file changes (add/remove/modify) introduced in this version",
    )
    ls.set_defaults(func=ds_list)

    get = subparsers.add_parser("get", help="Get a local copy of a dataset (default: read only cached folder)")
    get.add_argument(
        "--id",
        type=str,
        required=False,
        help="Previously created dataset id. Default: previously created/accessed dataset",
    )
    get.add_argument(
        "--copy",
        type=str,
        default=None,
        help="Get a writable copy of the dataset to a specific output folder",
    )
    get.add_argument(
        "--link",
        type=str,
        default=None,
        help="Create a soft link (not supported on Windows) to a read-only cached folder containing the dataset",
    )
    get.add_argument(
        "--part",
        type=int,
        default=None,
        help="Retrieve a partial copy of the dataset. Part number (0 to `num-parts`-1) of total parts --num-parts.",
    )
    get.add_argument(
        "--num-parts",
        type=int,
        default=None,
        help="Total number of parts to divide the dataset to. "
        "Notice minimum retrieved part is a single chunk in a dataset (or its parents)."
        "Example: Dataset gen4, with 3 parents, each with a single chunk, "
        "can be divided into 4 parts",
    )
    get.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If True, overwrite the target folder",
    )
    get.add_argument("--verbose", action="store_true", default=False, help="Verbose reporting")
    get.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of threads to get the files with. Defaults to the number of logical cores",
    )
    get.set_defaults(func=ds_get)

    args = parser.parse_args()
    args = restore_state(args)

    if args.command:
        args.func(args)
    else:
        parser.print_help()
    return 0


def ds_delete(args: Any) -> int:
    if args.id:
        print("Deleting dataset id {}".format(args.id))
    else:
        print("Deleting dataset with project={}, name={}, version={}".format(args.project, args.name, args.version))
    print_args(args)
    Dataset.delete(
        dataset_id=args.id,
        dataset_project=args.project,
        dataset_name=args.name,
        dataset_version=args.version,
        entire_dataset=args.entire_dataset,
        force=args.force,
    )
    print("Dataset(s) deleted")
    clear_state()
    return 0


def ds_rename(args: Any) -> int:
    print("Renaming dataset with project={}, name={} to {}".format(args.project, args.name, args.new_name))
    print_args(args)
    Dataset.rename(
        args.new_name,
        dataset_project=args.project,
        dataset_name=args.name,
    )
    print("Dataset(s) renamed")
    clear_state()
    return 0


def ds_move(args: Any) -> int:
    print("Moving dataset with project={}, name={} to {}".format(args.project, args.name, args.new_project))
    print_args(args)
    Dataset.move_to_project(
        args.new_project,
        dataset_project=args.project,
        dataset_name=args.name,
    )
    print("Dataset(s) moved")
    clear_state()
    return 0


def ds_verify(args: Any) -> None:
    print("Verify dataset id {}".format(args.id))
    check_null_id(args)
    print_args(args)
    ds = Dataset.get(dataset_id=args.id)
    files_error = ds.verify_dataset_hash(
        local_copy_path=args.folder or None,
        skip_hash=args.filesize,
        verbose=args.verbose,
    )
    if files_error:
        print("Dataset verification completed, {} errors found!".format(len(files_error)))
    else:
        print("Dataset verification completed successfully, no errors found.")


def ds_get(args: Any) -> int:
    print("Download dataset id {}".format(args.id))
    check_null_id(args)
    print_args(args)
    ds = Dataset.get(dataset_id=args.id)
    if args.overwrite:
        if args.copy:
            # noinspection PyBroadException
            try:
                shutil.rmtree(args.copy)
            except Exception:
                pass
            Path(args.copy).mkdir(parents=True, exist_ok=True)
        elif args.link:
            # noinspection PyBroadException
            try:
                shutil.rmtree(args.link)
            except Exception:
                pass
    if args.copy:
        ds_folder = args.copy
        ds.get_mutable_local_copy(
            target_folder=ds_folder,
            part=args.part,
            num_parts=args.num_parts,
            max_workers=args.max_workers,
        )
    else:
        if args.link:
            Path(args.link).mkdir(parents=True, exist_ok=True)
            # noinspection PyBroadException
            try:
                Path(args.link).rmdir()
            except Exception:
                try:
                    Path(args.link).unlink()
                except Exception:
                    raise ValueError("Target directory {} is not empty. Use --overwrite.".format(args.link))
        ds_folder = ds.get_local_copy(part=args.part, num_parts=args.num_parts, max_workers=args.max_workers)
        if args.link:
            os.symlink(ds_folder, args.link)
            ds_folder = args.link
    print("Dataset local copy available for files at: {}".format(ds_folder))
    return 0


def ds_list(args: Any) -> int:
    print("List dataset content: {}".format(args.id or (args.project, args.name)))
    print_args(args)
    ds = Dataset.get(
        dataset_id=args.id or None,
        dataset_project=args.project or None,
        dataset_name=args.name or None,
        dataset_version=args.version,
    )
    filters = args.filter if args.filter else [None]
    file_entries = ds.file_entries_dict
    link_entries = ds.link_entries_dict
    file_name_max_len, size_max_len, hash_max_len = 64, 10, 64
    files_cache = []
    for mask in filters:
        files = ds.list_files(dataset_path=mask, dataset_id=ds.id if args.modified else None)
        files_cache.append(files)
        for f in files:
            e = link_entries.get(f)
            if file_entries.get(f):
                e = file_entries[f]
            file_name_max_len = max(file_name_max_len, len(e.relative_path))
            size_max_len = max(size_max_len, len(str(e.size)))
            hash_max_len = max(hash_max_len, len(str(e.hash)))
    print("Listing dataset content")
    formatting = "{:" + str(file_name_max_len) + "} | {:" + str(size_max_len) + ",} | {:" + str(hash_max_len) + "}"
    print(formatting.replace(",", "").format("file name", "size", "hash"))
    print("-" * len(formatting.replace(",", "").format("-", "-", "-")))
    num_files = 0
    total_size = 0
    for files in files_cache:
        num_files += len(files)
        for f in files:
            e = link_entries.get(f)
            if file_entries.get(f):
                e = file_entries[f]
            print(formatting.format(e.relative_path, e.size, str(e.hash)))
            total_size += e.size
    print("Total {} files, {} bytes".format(num_files, total_size))
    return 0


def ds_squash(args: Any) -> int:
    print("Squashing datasets ids={} into target dataset named '{}'".format(args.ids, args.name))
    print_args(args)
    ds = Dataset.squash(dataset_name=args.name, dataset_ids=args.ids, output_url=args.storage or None)
    print("Squashing completed, new dataset created id={}".format(ds.id))
    return 0


def ds_search(args: Any) -> int:
    print("Search datasets")
    print_args(args)
    datasets = Dataset.list_datasets(
        dataset_project=args.project or None,
        partial_name=args.name or None,
        tags=args.tags or None,
        ids=args.ids or None,
        only_completed=not args.not_only_completed,
        recursive_project_search=not args.non_recursive_project_search,
    )
    projects_col_len, name_col_len, tags_col_len, created_col_len, id_col_len = (
        16,
        32,
        19,
        19,
        32,
    )
    for d in datasets:
        projects_col_len = max(projects_col_len, len(d["project"]))
        name_col_len = max(name_col_len, len(d["name"]))
        tags_col_len = max(tags_col_len, len(str(d["tags"] or [])[1:-1]))
        created_col_len = max(created_col_len, len(str(d["created"]).split(".")[0]))
        id_col_len = max(id_col_len, len(d["id"]))
    formatting = (
        "{:"
        + str(projects_col_len)
        + "} | {:"
        + str(name_col_len)
        + "} | {:"
        + str(tags_col_len)
        + "} | {:"
        + str(created_col_len)
        + "} | {:"
        + str(id_col_len)
        + "}"
    )
    print(formatting.format("project", "name", "version", "tags", "created", "id"))
    print("-" * len(formatting.format("-", "-", "-", "-", "-")))
    for d in datasets:
        print(
            formatting.format(
                d["project"],
                d["name"],
                d["version"] or "",
                str(d["tags"] or [])[1:-1],
                str(d["created"]).split(".")[0],
                d["id"],
            )
        )
    return 0


def ds_compare(args: Any) -> int:
    print("Comparing target dataset id {} with source dataset id {}".format(args.target, args.source))
    print_args(args)
    ds = Dataset.get(dataset_id=args.target)
    removed_files = ds.list_removed_files(dataset_id=args.source)
    modified_files = ds.list_modified_files(dataset_id=args.source)
    added_files = ds.list_added_files(dataset_id=args.source)
    if args.verbose:
        print("Removed files:")
        print("\n".join(removed_files))
        print("\nModified files:")
        print("\n".join(modified_files))
        print("\nAdded files:")
        print("\n".join(added_files))
        print("")
    print(
        "Comparison summary: {} files removed, {} files modified, {} files added".format(
            len(removed_files), len(modified_files), len(added_files)
        )
    )
    return 0


def ds_close(args: Any) -> int:
    print("Finalizing dataset id {}".format(args.id))
    check_null_id(args)
    print_args(args)
    ds = Dataset.get(dataset_id=args.id)
    if ds.is_dirty():
        if args.disable_upload:
            raise ValueError("Pending uploads, cannot finalize dataset. run `clearml-data upload`")
        # upload the files
        print("Pending uploads, starting dataset upload to {}".format(args.storage or ds.get_default_storage()))
        ds.upload(
            show_progress=True,
            verbose=args.verbose,
            output_url=args.storage or None,
            chunk_size=args.chunk_size or -1,
            max_workers=args.max_workers,
        )

    ds.finalize()
    print("Dataset closed and finalized")
    clear_state()
    return 0


def ds_publish(args: Any) -> int:
    print("Publishing dataset id {}".format(args.id))
    check_null_id(args)
    print_args(args)
    ds = Dataset.get(dataset_id=args.id)
    if not ds.is_final():
        raise ValueError("Cannot publish dataset. Please finalize it first, run `clearml-data close`")

    ds.publish()
    print("Dataset published")
    clear_state()  # just to verify the state is clear
    return 0


def ds_upload(args: Any) -> int:
    print("uploading local files to dataset id {}".format(args.id))
    check_null_id(args)
    print_args(args)
    ds = Dataset.get(dataset_id=args.id)
    ds.upload(
        verbose=args.verbose,
        output_url=args.storage or None,
        chunk_size=args.chunk_size or -1,
        max_workers=args.max_workers,
    )
    print("Dataset upload completed")
    return 0


def ds_remove(args: Any) -> int:
    print("Removing files/folder from dataset id {}".format(args.id))
    check_null_id(args)
    print_args(args)
    ds = Dataset.get(dataset_id=args.id)
    num_files = 0
    for file in args.files or []:
        num_files += ds.remove_files(dataset_path=file, recursive=not args.non_recursive, verbose=args.verbose)
    message = "{} file{} removed".format(num_files, "s" if num_files != 1 else "")
    print(message)
    return 0


def ds_sync(args: Any) -> int:
    dataset_created = False
    if args.parents or (args.project and args.name):
        args.id = ds_create(args)
        dataset_created = True

    print("Syncing dataset id {} to local folder {}".format(args.id, args.folder))
    check_null_id(args)
    print_args(args)
    ds = Dataset.get(dataset_id=args.id)
    removed, added, modified = ds.sync_folder(
        local_path=args.folder,
        dataset_path=args.dataset_folder or None,
        verbose=args.verbose,
    )

    print("Sync completed: {} files removed, {} added, {} modified".format(removed, added, modified))

    if not args.skip_close:
        if dataset_created and not removed and not added and not modified:
            print("Zero modifications on local copy, reverting dataset creation.")
            Dataset.delete(ds.id, force=True)
            return 0

        print("Finalizing dataset")
        if ds.is_dirty():
            # upload the files
            print("Pending uploads, starting dataset upload to {}".format(args.storage or ds.get_default_storage()))
            ds.upload(
                show_progress=True,
                verbose=args.verbose,
                output_url=args.storage or None,
                chunk_size=args.chunk_size or -1,
            )

        ds.finalize()
        print("Dataset closed and finalized")
        clear_state()

    return 0


def ds_add(args: Any) -> int:
    print("Adding files/folder/links to dataset id {}".format(args.id))
    check_null_id(args)
    print_args(args)
    ds = Dataset.get(dataset_id=args.id)
    num_files = 0
    for file in args.files or []:
        num_files += ds.add_files(
            path=file,
            recursive=not args.non_recursive,
            verbose=args.verbose,
            dataset_path=args.dataset_folder or None,
            wildcard=args.wildcard,
            max_workers=args.max_workers,
        )
    for link in args.links or []:
        num_files += ds.add_external_files(
            link,
            dataset_path=args.dataset_folder or None,
            recursive=not args.non_recursive,
            verbose=args.verbose,
            wildcard=args.wildcard,
            max_workers=args.max_workers,
        )
    message = "{} file{} added".format(num_files, "s" if num_files != 1 else "")
    print(message)
    return 0


def ds_create(args: Any) -> str:
    print("Creating a new dataset:")
    print_args(args)
    if args.output_uri:
        print("Warning: '--output-uri' is deprecated, use '--storage' instead")
    storage = args.storage or args.output_uri
    ds = Dataset.create(
        dataset_project=args.project,
        dataset_name=args.name,
        parent_datasets=args.parents,
        dataset_version=args.version,
        output_uri=storage,
    )
    if args.tags:
        ds.tags = ds.tags + args.tags
    print("New dataset created id={}".format(ds.id))
    clear_state({"id": ds.id})
    return ds.id


def ds_set_description(args: Any) -> int:
    check_null_id(args)
    print("Setting description '{}' to dataset {}".format(args.description, args.id))
    print_args(args)
    ds = Dataset.get(dataset_id=args.id)
    ds.set_description(args.description)
    return 0


def main() -> None:
    try:
        exit(cli())
    except KeyboardInterrupt:
        print("\nUser aborted")
    except Exception as ex:
        print("\nError: {}".format(ex))
        exit(1)


if __name__ == "__main__":
    main()
